import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path
import csv
import json
import socket

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import io
import base64
import tempfile

import numpy as np
import pickle
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import circuit_drawer
import tensorflow as tf



_CLASS_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

# ── Shared styles ──────────────────────────────────────────────────────────────

_IMAGE_CONTAINER = {"display": "flex", "justifyContent": "center",
                    "alignItems": "center", "overflow": "auto", "padding": "10px"}
_CIRCUIT_IMG     = {"maxWidth": "900px", "width": "100%", "height": "auto"}
_MODEL_IMG       = {"maxWidth": "400px", "width": "100%", "height": "auto"}
_GRAPH           = {"width": "100%", "height": "450px", "marginBottom": "30px"}
_SAMPLE          = {"width": "100%", "height": "450px", "marginBottom": "30px"}
_PAGE            = {"maxWidth": "1400px", "margin": "auto",
                    "padding": "20px", "fontFamily": "Arial"}
_SELECTOR        = {"width": "400px", "marginBottom": "30px",
                    "fontFamily": "Arial", "fontSize": "14px"}
_SELECTOR_ROW    = {"display": "flex", "alignItems": "center",
                    "gap": "16px", "marginBottom": "30px"}


# ══════════════════════════════════════════════════════════════════════════════
#  File readers  (stateless, receive run_dir explicitly)
# ══════════════════════════════════════════════════════════════════════════════

def _read_metadata(run_dir: Path) -> dict:
    with open(run_dir / "meta.json") as f:
        return json.load(f)

def _read_parameters(run_dir: Path) -> list:
    params = []
    with open(run_dir / "params.csv") as f:
        for row in csv.reader(f):
            if row:
                params.append([float(x) for x in row])
    return params

def _read_losses(run_dir: Path) -> list:
    losses = []
    with open(run_dir / "losses.csv") as f:
        for row in csv.DictReader(f):
            losses.append({"step": int(row["step"]),
                           "generator_loss": float(row["generator_loss"]),
                           "discriminator_loss": float(row["discriminator_loss"])})
    return losses

def _read_metrics(run_dir: Path) -> list:
    result = []
    with open(run_dir / "metrics.csv") as f:
        for row in csv.DictReader(f):
            row["step"] = int(row["step"])
            for k in row:
                if k != "step":
                    row[k] = float(row[k])
            result.append(row)
    return result

def _load_circuit(run_dir: Path):
    with open(run_dir / "generator_circuit.qasm", "rb") as f:
        return pickle.load(f)

def _sample_circuit(run_dir: Path, sampler: StatevectorSampler,
                    shots: int = 2**10) -> dict:
    params = _read_parameters(run_dir)
    if not params:
        return {}
    qc = _load_circuit(run_dir)
    qc_run = qc.copy()
    qc_run.measure_all()
    param_dict = dict(zip(qc.parameters, params[-1]))
    job = sampler.run([(qc_run, param_dict)], shots=shots)
    return job.result()[0].data.meas.get_counts()

def _sample_circuit_conditional(run_dir: Path, sampler: StatevectorSampler,
                                label: int, num_classes: int,
                                shots: int = 2**10) -> dict:
    params = _read_parameters(run_dir)
    if not params:
        return {}
    qc = _load_circuit(run_dir)
    num_qubits = qc.num_qubits
    encoding = QuantumCircuit(num_qubits)
    for qubit in label2bits(label, num_qubits):
        encoding.x(qubit)
    qc_run = encoding.compose(qc)
    qc_run.measure_all()
    param_dict = dict(zip(qc.parameters, params[-1]))
    job = sampler.run([(qc_run, param_dict)], shots=shots)
    return job.result()[0].data.meas.get_counts()


# ══════════════════════════════════════════════════════════════════════════════
#  Shared figure builders  (stateless)
# ══════════════════════════════════════════════════════════════════════════════

def _standardize(fig, height=450):
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=60, r=30, t=50, b=50),
                      xaxis_title="Epoch")
    return fig

def _build_metadata_table(run_dir: Path):
    meta = _read_metadata(run_dir)
    rows = [html.Tr([
        html.Td(str(k), style={"fontWeight": "bold", "padding": "4px 8px"}),
        html.Td(str(v), style={"padding": "4px 8px"})])
        for k, v in meta.items()]
    return html.Table(rows, style={"borderCollapse": "collapse", "width": "100%",
                                   "marginBottom": "20px", "border": "1px solid #ccc"})

def _build_circuit_figure(run_dir: Path):
    qc  = _load_circuit(run_dir)
    fig = circuit_drawer(qc, output="mpl")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    return html.Div(
        html.Img(src=f"data:image/png;base64,{img}", style=_CIRCUIT_IMG),
        style=_IMAGE_CONTAINER)

def _build_model_plot(run_dir: Path):
    model = tf.keras.models.load_model(run_dir / "discriminator_model.keras")
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        tf.keras.utils.plot_model(
            model, to_file=tmp.name, show_shapes=True,
            show_layer_names=True, rankdir="TB",
            expand_nested=True, show_layer_activations=True,
            show_trainable=True)
        with open(tmp.name, "rb") as f:
            img = base64.b64encode(f.read()).decode()
    return html.Div(
        html.Img(src=f"data:image/png;base64,{img}", style=_MODEL_IMG),
        style=_IMAGE_CONTAINER)

def _build_loss_figure(run_dir: Path):
    losses = _read_losses(run_dir)
    meta   = _read_metadata(run_dir)
    steps  = [l["step"] for l in losses]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=[l["generator_loss"] for l in losses],
                             mode="lines", name="Generator loss"))
    fig.add_trace(go.Scatter(x=steps, y=[l["discriminator_loss"] for l in losses],
                             mode="lines", name="Discriminator loss"))
    ref = 0.0 if meta.get("wasserstein") else np.log(2)
    txt = "0"  if meta.get("wasserstein") else "-log(1/2)"
    fig.add_hline(y=ref, line_dash="dot", line_color="gray",
                  annotation_text=txt, annotation_position="top right")
    fig.update_layout(title="Losses", yaxis_title="Loss")
    return _standardize(fig)

def _build_metrics_figure(run_dir: Path):
    met = _read_metrics(run_dir)
    if not met:
        return go.Figure()
    meta  = _read_metadata(run_dir)
    steps = [m["step"] for m in met]
    fig   = go.Figure()
    for key in met[0]:
        if key != "step":
            fig.add_trace(go.Scatter(x=steps, y=[m[key] for m in met],
                                     mode="lines", name=key))
    baseline = meta.get("baseline_js")
    if baseline:
        mean_js, std_js = baseline if isinstance(baseline, (list, tuple)) else (baseline, 0)
        fig.add_hline(y=mean_js, line_dash="dot", line_color="black",
                      annotation_text="baseline JS", annotation_position="top left")
        fig.add_hline(y=mean_js + std_js, line_dash="dot", line_color="gray")
        fig.add_hline(y=max(0., mean_js - std_js), line_dash="dot", line_color="gray")
    fig.update_layout(title="Metrics",
                      yaxis=dict(range=[0, 1]))
    return _standardize(fig)

def _build_param_heatmap(run_dir: Path):
    params = _read_parameters(run_dir)
    if not params:
        return go.Figure()
    z = np.mod(np.array(params).T, 2 * np.pi)
    colorscale = [[0.0, "yellow"], [0.25, "cyan"],
                  [0.75, "magenta"], [1.0, "yellow"]]
    fig = go.Figure(data=go.Heatmap(
        z=z, colorscale=colorscale, zsmooth=False, zmin=0, zmax=2 * np.pi,
        colorbar=dict(title="θ",
                      tickvals=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                      ticktext=["0", "π/2", "π", "3π/2", "2π"])))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Parameter heatmap", yaxis_title="Parameter")
    return _standardize(fig)

def _build_param_velocity_heatmap(run_dir: Path):
    params = _read_parameters(run_dir)
    if len(params) < 2:
        return go.Figure()
    delta = np.angle(np.exp(1j * np.diff(np.array(params), axis=0)))
    fig = go.Figure(data=go.Heatmap(
        z=np.abs(delta).T, colorscale="Inferno",
        colorbar=dict(title="|Δθ|"), zsmooth=False, zmin=0))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Parameter velocity heatmap", yaxis_title="Parameter")
    return _standardize(fig)

def _build_sample_plot(run_dir: Path, sampler: StatevectorSampler):
    qc     = _load_circuit(run_dir)
    n      = qc.num_qubits
    states = [format(b, f'0{n}b') for b in range(2 ** n)]
    sample = _sample_circuit(run_dir, sampler)
    values = [sample.get(s, 0) for s in states]
    total  = sum(values) or 1
    fig = go.Figure()
    fig.add_bar(x=states, y=[v / total for v in values])
    fig.update_layout(
        title="Generated sample", xaxis_title="Bins",
        yaxis_title="Probability", template="plotly_white",
        yaxis=dict(range=[0, 1]),
        xaxis=dict(tickmode="linear", tick0=0, dtick=max(1, 2**n // 16)))
    return _standardize(fig)

def _build_conditional_metrics_figure(run_dir: Path, num_classes: int):
    met = _read_metrics(run_dir)
    if not met:
        return go.Figure()
    meta  = _read_metadata(run_dir)
    steps = [m["step"] for m in met]
    fig   = go.Figure()
    for k in range(num_classes):
        color   = _CLASS_COLORS[k % len(_CLASS_COLORS)]
        js_key  = f"js_class_{k}"
        fid_key = f"fidelity_class_{k}"
        if js_key in met[0]:
            fig.add_trace(go.Scatter(x=steps, y=[m[js_key] for m in met],
                                     mode="lines", name=f"JS class {k}",
                                     line=dict(color=color, dash="dot"), opacity=0.6))
        if fid_key in met[0]:
            fig.add_trace(go.Scatter(x=steps, y=[m[fid_key] for m in met],
                                     mode="lines", name=f"Fidelity class {k}",
                                     line=dict(color=color), opacity=0.55,
                                     yaxis="y2"))
    for key, label, style in [
        ("js_aggregate",     "JS aggregate",          dict(color="black", width=2)),
        ("js_aggregate_avg", "JS aggregate (avg)",    dict(color="black", width=2, dash="dash")),
    ]:
        if key in met[0]:
            fig.add_trace(go.Scatter(x=steps, y=[m[key] for m in met],
                                     mode="lines", name=label, line=style))
    if "fidelity_aggregate" in met[0]:
        fig.add_trace(go.Scatter(x=steps, y=[m["fidelity_aggregate"] for m in met],
                                 mode="lines", name="Fidelity aggregate",
                                 line=dict(color="darkblue", width=2), yaxis="y2"))
    baseline = meta.get("baseline_js")
    if baseline:
        mean_js, std_js = baseline if isinstance(baseline, (list, tuple)) else (baseline, 0)
        fig.add_hline(y=mean_js, line_dash="dot", line_color="gray",
                      annotation_text="baseline JS", annotation_position="top left")
        fig.add_hline(y=mean_js + std_js, line_dash="dot", line_color="lightgray")
        fig.add_hline(y=max(0., mean_js - std_js), line_dash="dot", line_color="lightgray")
    fig.update_layout(
        title="Conditional metrics",
        yaxis=dict(title="Jensen-Shannon divergence"),
        yaxis2=dict(title="Fidelity", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white", height=500,
        margin=dict(l=60, r=60, t=60, b=50))
    return fig

def _build_class_sample_plots(run_dir: Path, sampler: StatevectorSampler,
                               num_classes: int):
    qc     = _load_circuit(run_dir)
    n      = qc.num_qubits
    states = [format(b, f'0{n}b') for b in range(2 ** n)]
    graphs = []
    for k in range(num_classes):
        sample = _sample_circuit_conditional(run_dir, sampler, k, num_classes)
        values = [sample.get(s, 0) for s in states]
        total  = sum(values) or 1
        color  = _CLASS_COLORS[k % len(_CLASS_COLORS)]
        fig = go.Figure()
        fig.add_bar(x=states, y=[v / total for v in values], marker_color=color)
        fig.update_layout(
            title=f"Generated sample — class {k}",
            xaxis_title="Bins", yaxis_title="Probability",
            template="plotly_white", yaxis=dict(range=[0, 1]),
            xaxis=dict(tickmode="linear", tick0=0, dtick=max(1, 2**n // 16)),
            height=380, margin=dict(l=50, r=20, t=50, b=40))
        graphs.append(dcc.Graph(id=f"sample-class-{k}", figure=fig,
                                style={"width": "100%", "marginBottom": "20px"}))
    return graphs


# ══════════════════════════════════════════════════════════════════════════════
#  Run selector helper
# ══════════════════════════════════════════════════════════════════════════════

def _list_runs(output_dir: Path) -> list:
    '''Return sorted list of run directory names found in output_dir.'''
    if not output_dir.exists():
        return []
    return sorted([d.name for d in output_dir.iterdir()
                   if d.is_dir() and (d / "meta.json").exists()],
                  reverse=True)   # most recent first


# ══════════════════════════════════════════════════════════════════════════════
#  TrainingDashboard  —  unconditional, run-picker UI
# ══════════════════════════════════════════════════════════════════════════════

class TrainingDashboard:
    '''
    Dashboard for unconditional QGAN runs.

    No run_dir is required at construction — the user selects a run from a
    dropdown inside the UI.  The output directory is scanned for available
    runs every time the dropdown is opened (via a Refresh button).

    Parameters
    ----------
    output_dir   : path to the directory that contains run_* subdirectories.
                   Defaults to "./output".
    refresh_rate : live-update interval in seconds (default 2).

    Usage
    -----
        from qgan_lamarr import TrainingDashboard
        TrainingDashboard().run()
        # or point at a custom output location:
        TrainingDashboard("/scratch/myproject/output").run()
    '''

    def __init__(self, output_dir: str = "./output", refresh_rate: float = 2):
        self.output_dir   = Path(output_dir)
        self.refresh_rate = refresh_rate
        self.sampler      = StatevectorSampler()
        self.app          = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _run_selector(self):
        runs = _list_runs(self.output_dir)
        options = [{"label": r, "value": r} for r in runs]
        return html.Div([
            dcc.Dropdown(
                id="run-selector",
                options=options,
                value=runs[0] if runs else None,
                placeholder="Select a run...",
                clearable=False,
                style=_SELECTOR),
            html.Button("↻ Refresh list", id="refresh-btn",
                        style={"height": "36px", "cursor": "pointer"}),
        ], style=_SELECTOR_ROW)

    def _setup_layout(self):
        self.app.layout = html.Div([
            html.H1("QGAN Training Dashboard", style={"marginBottom": "20px"}),
            self._run_selector(),

            # Static panels (rebuilt when run changes)
            html.Div(id="static-panels"),

            # Live-updating graphs
            dcc.Graph(id="loss-graph",        style=_GRAPH),
            dcc.Graph(id="metrics-graph",     style=_GRAPH),
            dcc.Graph(id="param-heatmap",     style=_GRAPH),
            dcc.Graph(id="param-vel-heatmap", style=_GRAPH),
            dcc.Graph(id="generated-sample",  style=_SAMPLE),

            dcc.Interval(id="interval",
                         interval=int(self.refresh_rate * 1000), n_intervals=0),
        ], style=_PAGE)

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _setup_callbacks(self):
        app = self.app

        # Refresh the run list when the button is clicked
        @app.callback(
            Output("run-selector", "options"),
            Output("run-selector", "value"),
            Input("refresh-btn", "n_clicks"),
            State("run-selector", "value"),
            prevent_initial_call=True)
        def refresh_runs(_, current):
            runs    = _list_runs(self.output_dir)
            options = [{"label": r, "value": r} for r in runs]
            # Keep current selection if it still exists, else pick the newest
            value = current if current in runs else (runs[0] if runs else None)
            return options, value

        # Rebuild static panels when the run changes
        @app.callback(
            Output("static-panels", "children"),
            Input("run-selector", "value"))
        def update_static(run_name):
            if not run_name:
                return html.P("No run selected.", style={"color": "gray"})
            run_dir = self.output_dir / run_name
            try:
                return [
                    html.Details([html.Summary("Run metadata"),
                                  _build_metadata_table(run_dir)],
                                 style={"marginBottom": "20px"}),
                    html.Details([html.Summary("Generator circuit"),
                                  _build_circuit_figure(run_dir)],
                                 style={"marginBottom": "20px"}),
                    html.Details([html.Summary("Discriminator model"),
                                  _build_model_plot(run_dir)],
                                 style={"marginBottom": "20px"}),
                ]
            except Exception as e:
                return html.P(f"Error loading run: {e}", style={"color": "red"})

        # Live-update graphs every interval tick
        @app.callback(
            Output("loss-graph",        "figure"),
            Output("metrics-graph",     "figure"),
            Output("param-heatmap",     "figure"),
            Output("param-vel-heatmap", "figure"),
            Output("generated-sample",  "figure"),
            Input("interval",           "n_intervals"),
            Input("run-selector",       "value"))
        def update_graphs(_, run_name):
            empty = go.Figure()
            if not run_name:
                return empty, empty, empty, empty, empty
            run_dir = self.output_dir / run_name
            try:
                return (
                    _build_loss_figure(run_dir),
                    _build_metrics_figure(run_dir),
                    _build_param_heatmap(run_dir),
                    _build_param_velocity_heatmap(run_dir),
                    _build_sample_plot(run_dir, self.sampler),
                )
            except Exception as e:
                err = go.Figure()
                err.add_annotation(text=str(e), xref="paper", yref="paper",
                                   x=0.5, y=0.5, showarrow=False)
                return err, err, err, err, err

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self, port=None):
        if port is None:
            with socket.socket() as s:
                s.bind(('', 0))
                port = s.getsockname()[1]
        ip = socket.gethostbyname(socket.gethostname())
        print(f"Output directory: {self.output_dir}")
        print(f"Dashboard URL:    http://{ip}:{port}")
        self.app.run(host="0.0.0.0", debug=False, port=port)



# ══════════════════════════════════════════════════════════════════════════════
#  XMapCQGAN dashboard figure builders  (stateless)
# ══════════════════════════════════════════════════════════════════════════════

def _xmap_sample_circuit(run_dir: Path, sampler: StatevectorSampler,
                         label: int, shots: int = 2**10) -> dict:
    '''
    Reproduce XMapCQGAN.cond_generator_eval:
    apply X gates from the label's entry in _bins, compose with the ansatz,
    then sample with the latest saved parameters.
    '''
    params = _read_parameters(run_dir)
    if not params:
        return {}
    qc   = _load_circuit(run_dir)
    n    = qc.num_qubits
    bins = [format(b, f'0{n}b') for b in range(2 ** n)]

    # Reproduce prepare_xmap indexing from models.py:
    # qubit i gets X if bit (n-1-i) of the label's bitstring is '1'
    bitstr   = bins[label]
    encoding = QuantumCircuit(n)
    for xbit in range(n):
        if bitstr[n - 1 - xbit] == '1':
            encoding.x(xbit)

    qc_run = encoding.compose(qc)
    qc_run.measure_all()
    param_dict = dict(zip(qc.parameters, params[-1]))
    job = sampler.run([(qc_run, param_dict)], shots=shots)
    return job.result()[0].data.meas.get_counts()


def _build_xmap_metrics_figure(run_dir: Path, num_classes: int):
    '''
    Metrics figure for XMapCQGAN.

    Matches the key scheme written by XMapCQGAN.cond_compute_metrics and
    total_compute_metrics:
        per-class : jensen_shannon_c{k}, fidelity_c{k}          (dotted, low opacity)
        aggregate : jensen_shannon, jensen_shannon_avg           (solid black, left axis)
                    fidelity, fidelity_avg                       (solid darkblue, right axis)
    '''
    met = _read_metrics(run_dir)
    if not met:
        return go.Figure()
    meta  = _read_metadata(run_dir)
    steps = [m['step'] for m in met]
    fig   = go.Figure()

    # Per-class JS — left axis, dotted
    for k in range(num_classes):
        key   = f'jensen_shannon_c{k}'
        color = _CLASS_COLORS[k % len(_CLASS_COLORS)]
        if key in met[0]:
            fig.add_trace(go.Scatter(
                x=steps, y=[m[key] for m in met],
                mode='lines', name=f'JS c{k}',
                line=dict(color=color, dash='dot'), opacity=0.55))

    # Per-class fidelity — right axis, solid
    for k in range(num_classes):
        key   = f'fidelity_c{k}'
        color = _CLASS_COLORS[k % len(_CLASS_COLORS)]
        if key in met[0]:
            fig.add_trace(go.Scatter(
                x=steps, y=[m[key] for m in met],
                mode='lines', name=f'Fidelity c{k}',
                line=dict(color=color), opacity=0.45,
                yaxis='y2'))

    # Aggregate JS — left axis, bold
    for key, label, style in [
        ('jensen_shannon',     'JS aggregate',       dict(color='black', width=2)),
        ('jensen_shannon_avg', 'JS aggregate (avg)', dict(color='black', width=2, dash='dash')),
    ]:
        if key in met[0]:
            fig.add_trace(go.Scatter(
                x=steps, y=[m[key] for m in met],
                mode='lines', name=label, line=style))

    # Aggregate fidelity — right axis, bold
    for key, label, style in [
        ('fidelity',     'Fidelity aggregate',       dict(color='darkblue', width=2)),
        ('fidelity_avg', 'Fidelity aggregate (avg)', dict(color='darkblue', width=2, dash='dash')),
    ]:
        if key in met[0]:
            fig.add_trace(go.Scatter(
                x=steps, y=[m[key] for m in met],
                mode='lines', name=label, line=style,
                yaxis='y2'))

    # Baseline JS band
    baseline = meta.get('baseline_js')
    if baseline:
        mean_js, std_js = baseline if isinstance(baseline, (list, tuple)) else (baseline, 0)
        fig.add_hline(y=mean_js, line_dash='dot', line_color='gray',
                      annotation_text='baseline JS', annotation_position='top left')
        fig.add_hline(y=mean_js + std_js, line_dash='dot', line_color='lightgray')
        fig.add_hline(y=max(0., mean_js - std_js), line_dash='dot', line_color='lightgray')

    fig.update_layout(
        title='Conditional metrics — XMapCQGAN',
        xaxis_title='Epoch',
        yaxis=dict(title='Jensen-Shannon divergence', range=[0, 1]),
        yaxis2=dict(title='Fidelity', overlaying='y', side='right', range=[0, 1]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white', height=520,
        margin=dict(l=60, r=60, t=70, b=50))
    return fig


def _build_xmap_class_samples(run_dir: Path, sampler: StatevectorSampler,
                               num_classes: int) -> list:
    '''
    2-column grid of bar charts, one per class.
    Each chart title shows the input basis state  |bitstring>
    so it's immediately clear which computational state the generator started from.
    '''
    qc     = _load_circuit(run_dir)
    n      = qc.num_qubits
    bins   = [format(b, f'0{n}b') for b in range(2 ** n)]
    states = bins                            # x-axis labels = all output bins
    dtick  = max(1, 2**n // 16)

    cards = []
    for k in range(num_classes):
        sample = _xmap_sample_circuit(run_dir, sampler, k)
        values = [sample.get(s, 0) for s in states]
        total  = sum(values) or 1
        color  = _CLASS_COLORS[k % len(_CLASS_COLORS)]

        fig = go.Figure()
        fig.add_bar(x=states, y=[v / total for v in values], marker_color=color)
        fig.update_layout(
            title=f'Class {k}  —  input |{bins[k]}>',
            xaxis_title='Output bins', yaxis_title='Probability',
            template='plotly_white',
            yaxis=dict(range=[0, 1]),
            xaxis=dict(tickmode='linear', tick0=0, dtick=dtick),
            height=320,
            margin=dict(l=45, r=15, t=45, b=40),
            showlegend=False)

        # 2-column layout: even-indexed cards get right margin, odd get none
        cards.append(html.Div(
            dcc.Graph(figure=fig, id=f'xmap-sample-c{k}'),
            style={
                'width': '48%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'marginBottom': '16px',
                'marginRight': '2%' if k % 2 == 0 else '0',
            }))
    return cards


# ══════════════════════════════════════════════════════════════════════════════
#  XMapCQGANDashboard
# ══════════════════════════════════════════════════════════════════════════════

class XMapCQGANDashboard(TrainingDashboard):
    '''
    Dashboard for XMapCQGAN runs.

    Inherits from TrainingDashboard:
        run-selector dropdown, refresh button, static panels (metadata /
        circuit / discriminator), loss graph, parameter heatmaps, live
        interval updates, random-port binding, cluster-friendly IP printing.

    Overrides / adds:
        - Metrics figure  matching XMapCQGAN key scheme
          (jensen_shannon_c{k}, fidelity_c{k}, jensen_shannon, fidelity, …)
        - Per-class sample grid  in a 2-column layout, each chart labelled
          with the input basis state  |bitstring>  so the conditioning is
          immediately visible.
        - Removes the unconditional "generated-sample" graph (not meaningful
          without specifying a class).

    Usage
    -----
        from qgan_lamarr import XMapCQGANDashboard
        XMapCQGANDashboard("./output").run()
    '''

    def _setup_layout(self):
        self.app.layout = html.Div([
            html.H1('XMapCQGAN Training Dashboard',
                    style={'marginBottom': '20px'}),
            self._run_selector(),

            html.Div(id='static-panels'),

            dcc.Graph(id='loss-graph',        style=_GRAPH),
            dcc.Graph(id='metrics-graph',     style={**_GRAPH, 'height': '520px'}),
            dcc.Graph(id='param-heatmap',     style=_GRAPH),
            dcc.Graph(id='param-vel-heatmap', style=_GRAPH),

            html.H3('Generated samples per class',
                    style={'marginTop': '10px', 'marginBottom': '12px',
                           'fontFamily': 'Arial'}),
            html.Div(id='class-sample-container'),

            dcc.Interval(id='interval',
                         interval=int(self.refresh_rate * 1000), n_intervals=0),
        ], style=_PAGE)

    def _setup_callbacks(self):
        app = self.app

        @app.callback(
            Output('run-selector', 'options'),
            Output('run-selector', 'value'),
            Input('refresh-btn',   'n_clicks'),
            State('run-selector',  'value'),
            prevent_initial_call=True)
        def refresh_runs(_, current):
            runs    = _list_runs(self.output_dir)
            options = [{'label': r, 'value': r} for r in runs]
            value   = current if current in runs else (runs[0] if runs else None)
            return options, value

        @app.callback(
            Output('static-panels', 'children'),
            Input('run-selector',   'value'))
        def update_static(run_name):
            if not run_name:
                return html.P('No run selected.', style={'color': 'gray'})
            run_dir = self.output_dir / run_name
            try:
                return [
                    html.Details([html.Summary('Run metadata'),
                                  _build_metadata_table(run_dir)],
                                 style={'marginBottom': '20px'}),
                    html.Details([html.Summary('Generator ansatz circuit'),
                                  _build_circuit_figure(run_dir)],
                                 style={'marginBottom': '20px'}),
                    html.Details([html.Summary('Discriminator model'),
                                  _build_model_plot(run_dir)],
                                 style={'marginBottom': '20px'}),
                ]
            except Exception as e:
                return html.P(f'Error loading run: {e}', style={'color': 'red'})

        @app.callback(
            Output('loss-graph',             'figure'),
            Output('metrics-graph',          'figure'),
            Output('param-heatmap',          'figure'),
            Output('param-vel-heatmap',      'figure'),
            Output('class-sample-container', 'children'),
            Input('interval',                'n_intervals'),
            Input('run-selector',            'value'))
        def update_graphs(_, run_name):
            empty = go.Figure()
            if not run_name:
                return empty, empty, empty, empty, []
            run_dir = self.output_dir / run_name
            try:
                meta        = _read_metadata(run_dir)
                num_classes = int(meta.get('num_classes', 2))
                return (
                    _build_loss_figure(run_dir),
                    _build_xmap_metrics_figure(run_dir, num_classes),
                    _build_param_heatmap(run_dir),
                    _build_param_velocity_heatmap(run_dir),
                    _build_xmap_class_samples(run_dir, self.sampler, num_classes),
                )
            except Exception as e:
                err = go.Figure()
                err.add_annotation(text=str(e), xref='paper', yref='paper',
                                   x=0.5, y=0.5, showarrow=False)
                return err, err, err, err, [
                    html.P(f'Error: {e}', style={'color': 'red'})]