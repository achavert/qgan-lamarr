import os
import csv
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Dict

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler



'''
Sample format converters
'''
def dict2sample(_dict):
    '''
        Converts dictionary to an array of samples.    
    '''
    samples = []
    for bitstring, count in _dict.items():
        samples.extend([bitstring] * count)
    samples = np.array(samples)
    return np.array([int(s, 2) for s in samples])

def dict2vector(sample: dict, bins) -> np.ndarray:
    '''
        Converts sample dictionary to probability vector.    
    '''
    total = sum(sample.values())
    vector = np.array([sample.get(b, 0) / total for b in bins], dtype = np.float32)
    return vector


def evaluate_model(run_dir: str, real_dist: Callable, shots: int = 1024, n_reps: int = 20) -> Dict:
    """
    Evalúa un modelo entrenado cargando su configuración desde el FileManager.
    
    Args:
        run_dir (str): Ruta al directorio de salida (ej. './output/run_2026...').
        real_dist (Callable): Función de la distribución real.
        shots (int): Número de shots para cada muestreo.
        n_reps (int): Número de repeticiones para calcular las incertidumbres.
        
    Returns:
        Dict: Diccionario con los metadatos, métricas baseline y métricas del modelo.
    """
    # Importación local para evitar dependencias circulares con metrics.py
    from .metrics import jensen_shannon, fidelity
    
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"No se encontró el directorio: {run_dir}")

    # 1. Cargar Metadatos
    with open(run_path / "meta.json", "r") as f:
        meta = json.load(f)

    # 2. Cargar Parámetros (última iteración)
    params = []
    with open(run_path / "params.csv", "r") as f:
        for row in csv.reader(f):
            if row:
                params.append([float(x) for x in row])
    final_weights = params[-1]

    # 3. Detectar Tipo de Modelo y Cargar Circuitos
    is_conditional = 'num_classes' in meta
    num_classes = int(meta.get('num_classes', 1))
    num_qubits = int(np.log2(meta['bins']))
    bins_str = [format(b, f'0{num_qubits}b') for b in range(meta['bins'])]
    
    run_type = 'qgan'
    if is_conditional:
        if (run_path / "xmap.pkl").exists():
            run_type = 'xmap'
        else:
            run_type = 'qcgan'

    with open(run_path / "generator_circuit.qasm", "rb") as f:
        circuit_obj = pickle.load(f)

    xmap = None
    if run_type == 'xmap':
        with open(run_path / "xmap.pkl", "rb") as f:
            xmap = pickle.load(f)

    sampler = StatevectorSampler()

    # --- Funciones auxiliares de muestreo ---
    def sample_model(c=0):
        if run_type == 'qgan':
            qc = circuit_obj.copy()
        elif run_type == 'xmap':
            qc = xmap[c].compose(circuit_obj, range(num_qubits))
        elif run_type == 'qcgan':
            qc = QuantumCircuit(num_qubits)
            for key in circuit_obj.schedule:
                if 'X_' in key:
                    qc = qc.compose(circuit_obj.schedule[key][c])
                elif 'G_' in key:
                    qc = qc.compose(circuit_obj.schedule[key])
        
        qc.measure_all()
        param_dict = dict(zip(qc.parameters, final_weights))
        job = sampler.run([(qc, param_dict)], shots=shots)
        return job.result()[0].data.meas.get_counts()

    def sample_real(c=0):
        if is_conditional:
            return real_dist(c, shots, meta['bins'])
        else:
            return real_dist(shots, meta['bins'])

    # --- Cálculo de Métricas ---
    baseline_js_list, baseline_fid_list = [], []
    model_js_list, model_fid_list = [], []

    real_vectors = {c: [] for c in range(num_classes)}
    model_vectors = {c: [] for c in range(num_classes)}

    print(f"Evaluando modelo ({run_type}) con {n_reps} repeticiones...")
    for _ in range(n_reps):
        rep_b_js, rep_b_fid = 0.0, 0.0
        rep_m_js, rep_m_fid = 0.0, 0.0

        for c in range(num_classes):
            real_1 = sample_real(c)
            real_2 = sample_real(c)
            model_1 = sample_model(c)

            # Guardar para la gráfica
            real_vectors[c].append(dict2vector(real_1, bins_str))
            model_vectors[c].append(dict2vector(model_1, bins_str))

            # Calcular iteración (promediado por clase si es condicional)
            weight = 1.0 / num_classes
            rep_b_js += jensen_shannon(real_1, real_2, bins_str) * weight
            rep_b_fid += fidelity(real_1, real_2, bins_str) * weight
            
            rep_m_js += jensen_shannon(real_1, model_1, bins_str) * weight
            rep_m_fid += fidelity(real_1, model_1, bins_str) * weight

        baseline_js_list.append(rep_b_js)
        baseline_fid_list.append(rep_b_fid)
        model_js_list.append(rep_m_js)
        model_fid_list.append(rep_m_fid)

    # Empaquetar resultados
    results = {
        "metadata": meta,
        "baseline": {
            "jensen_shannon": {"mean": float(np.mean(baseline_js_list)), "std": float(np.std(baseline_js_list))},
            "fidelity": {"mean": float(np.mean(baseline_fid_list)), "std": float(np.std(baseline_fid_list))}
        },
        "model": {
            "jensen_shannon": {"mean": float(np.mean(model_js_list)), "std": float(np.std(model_js_list))},
            "fidelity": {"mean": float(np.mean(model_fid_list)), "std": float(np.std(model_fid_list))}
        }
    }

    # --- Generación de Gráficas ---
    fig, axes = plt.subplots(1, num_classes, figsize=(6 * num_classes, 5), squeeze=False)
    axes = axes.flatten()
    x_pos = np.arange(meta['bins'])
    width = 0.35

    for c in range(num_classes):
        real_arr = np.array(real_vectors[c])
        model_arr = np.array(model_vectors[c])

        real_mean, real_std = np.mean(real_arr, axis=0), np.std(real_arr, axis=0)
        model_mean, model_std = np.mean(model_arr, axis=0), np.std(model_arr, axis=0)

        axes[c].bar(x_pos - width/2, real_mean, width, yerr=real_std, capsize=3, label='Real', alpha=0.7, color='gray')
        axes[c].bar(x_pos + width/2, model_mean, width, yerr=model_std, capsize=3, label='Generado', alpha=0.8, color='#4a90d9')

        title = f"Distribución Clase {c}" if is_conditional else "Distribución Generada vs Real"
        axes[c].set_title(title)
        axes[c].set_xticks(x_pos)
        axes[c].set_xticklabels(bins_str, rotation=90 if num_qubits > 3 else 0)
        axes[c].set_ylabel("Probabilidad")
        axes[c].legend()
        axes[c].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('plot.png')
    plt.show()

    return results
        




