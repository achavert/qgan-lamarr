# Quantum Generative Adversarial Networks for Lamarr fast-simulation at LHCb

## Installation
```bash
git clone <repo>
cd qgan-lamarr
pip install -e .          
```

## Usage
```python
from qgan import QGAN, TrainingDashboard, MinMaxBinning, SingleGaussian

# Define your real distribution callable
num_qubits = 3
nbins = 2**num_qubits

def real_dist(nshots, dim):
    data = SingleGaussian(mean=0.0, sd=1.0, shots=nshots)
    return MinMaxBinning(data, dim)


# Define quantum ansatz generator
from qiskit.circuit import ParameterVector

reps = 2
qc = QuantumCircuit(num_qubits)
qc.h(range(num_qubits))
theta = ParameterVector("θ", length = (reps + 1) * num_qubits * 2)
p = 0
for q in range(num_qubits):
        qc.ry(theta[q], q)
        p += 1
for r in range(reps):

    for q in range(num_qubits):
        qc.cz(q%num_qubits, (q + 1)%num_qubits)
    for q in range(num_qubits):
        qc.ry(theta[p], q)
        p += 1

# Define classical DNN discriminator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, LeakyReLU

discriminator = Sequential([
    InputLayer(shape=(nbins,)),
    Dense(50),
    LeakyReLU(),
    Dense(1, activation='linear') 
])

# Build generator circuit and discriminator model and fit:
model = QGAN(num_qubits = num_qubits,
             generator = qc,
             discriminator = discriminator,
             real_dist = real_dist)
model.fit(epochs=1000, shots=2**10, opt='ADAM_PSR', lr=1e-3, manager=True)

# Monitor training
dashboard = TrainingDashboard(run_dir="./output/run_<timestamp>")
dashboard.run()
```
