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

def _list_runs(root: Path) -> list[dict]:
    '''
    Recursively scan root for any directory containing meta.json.
    Returns a list of dicts suitable for a Dash dropdown:
        label : display string showing path relative to root
        value : path string relative to root  (use root / value to resolve)
    Sorted newest-first by directory name.
    '''
    if not root.exists():
        return []
    hits = sorted(
        [p.parent for p in root.rglob("meta.json") if p.parent.is_dir()],
        key=lambda p: p.name,
        reverse=True)
    return [{"label": str(p.relative_to(root)), "value": str(p.relative_to(root))}
            for p in hits]


def _detect_run_type(run_dir: Path) -> str:
    '''
    Read meta.json and return  'xmap'  if num_classes is present,
    otherwise  'qgan'.
    '''
    try:
        meta = _read_metadata(run_dir)
        return 'xmap' if 'num_classes' in meta else 'qgan'
    except Exception:
        return 'qgan'


# ══════════════════════════════════════════════════════════════════════════════
#  QGANDashboard  —  unified dashboard for QGAN and XMapQCGAN runs
# ══════════════════════════════════════════════════════════════════════════════

class QGANDashboard:
    '''
    Unified dashboard for QGAN and XMapQCGAN runs.

    Scans the given root directory recursively for any subdirectory containing
    a  meta.json  file — so any folder structure is supported.  The dropdown
    label shows the path relative to root, making it easy to navigate runs
    stored in subdirectories (e.g.  muons/run_20260519_120000).

    On run selection, meta.json is read to detect the run type:
        - 'num_classes' present  →  XMapQCGAN  (conditional graphs + per-class
          sample grid)
        - absent                  →  QGAN        (unconditional graphs + single
          sample plot)

    The layout contains both panel sets; type-detection controls which one is
    shown and which is hidden via a hidden 'run-type' Store.

    Parameters
    ----------
    root         : root directory to scan recursively. Defaults to '.'.
    refresh_rate : live-update interval in seconds (default 2).

    Usage
    -----
        from qgan_lamarr import QGANDashboard
        QGANDashboard(".").run()
        QGANDashboard("/scratch/myproject").run()
    '''

    def __init__(self, root: str = ".", refresh_rate: float = 2):
        self.root         = Path(root)
        self.refresh_rate = refresh_rate
        self.sampler      = StatevectorSampler()
        self.app          = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    # ── Run selector ───────────────────────────────────────────────────────────

    def _run_selector(self):
        runs = _list_runs(self.root)
        return html.Div([
            dcc.Dropdown(
                id="run-selector",
                options=runs,
                value=runs[0]["value"] if runs else None,
                placeholder="Select a run...",
                clearable=False,
                style=_SELECTOR),
            html.Button("↻ Refresh list", id="refresh-btn",
                        style={"height": "36px", "cursor": "pointer"}),
        ], style=_SELECTOR_ROW)

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _setup_layout(self):
        self.app.layout = html.Div([
            html.H1("QGAN Dashboard", style={"marginBottom": "20px"}),

            self._run_selector(),

            # Run-type store — written by update_static, read by update_graphs
            dcc.Store(id="run-type", data="qgan"),

            # Static panels: metadata, circuit, discriminator
            html.Div(id="static-panels"),

            # Shared graphs (both run types)
            dcc.Graph(id="loss-graph",        style=_GRAPH),
            dcc.Graph(id="metrics-graph",     style={**_GRAPH, "height": "520px"}),
            dcc.Graph(id="param-heatmap",     style=_GRAPH),
            dcc.Graph(id="param-vel-heatmap", style=_GRAPH),

            # QGAN-only: single sample plot
            html.Div(id="sample-panel",
                     children=[dcc.Graph(id="generated-sample", style=_SAMPLE)]),

            # XMapQCGAN-only: per-class sample grid
            html.Div(id="class-sample-panel", children=[
                html.H3("Generated samples per class",
                        style={"marginTop": "10px", "marginBottom": "12px"}),
                html.Div(id="class-sample-container"),
            ]),

            dcc.Interval(id="interval",
                         interval=int(self.refresh_rate * 1000), n_intervals=0),
        ], style=_PAGE)

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _setup_callbacks(self):
        app = self.app

        # ── Refresh run list ───────────────────────────────────────────────────
        @app.callback(
            Output("run-selector", "options"),
            Output("run-selector", "value"),
            Input("refresh-btn",   "n_clicks"),
            State("run-selector",  "value"),
            prevent_initial_call=True)
        def refresh_runs(_, current):
            runs   = _list_runs(self.root)
            values = [r["value"] for r in runs]
            value  = current if current in values else (values[0] if values else None)
            return runs, value

        # ── Static panels + run-type detection ────────────────────────────────
        @app.callback(
            Output("static-panels", "children"),
            Output("run-type",      "data"),
            Input("run-selector",   "value"))
        def update_static(run_value):
            if not run_value:
                return html.P("No run selected.", style={"color": "gray"}), "qgan"
            run_dir  = self.root / run_value
            run_type = _detect_run_type(run_dir)
            circuit_label = ("Generator ansatz circuit"
                             if run_type == "xmap" else "Generator circuit")
            try:
                panels = [
                    html.Details([html.Summary("Run metadata"),
                                  _build_metadata_table(run_dir)],
                                 style={"marginBottom": "20px"}),
                    html.Details([html.Summary(circuit_label),
                                  _build_circuit_figure(run_dir)],
                                 style={"marginBottom": "20px"}),
                    html.Details([html.Summary("Discriminator model"),
                                  _build_model_plot(run_dir)],
                                 style={"marginBottom": "20px"}),
                ]
            except Exception as e:
                panels = html.P(f"Error loading run: {e}", style={"color": "red"})
            return panels, run_type

        # ── Live graphs ────────────────────────────────────────────────────────
        @app.callback(
            Output("loss-graph",              "figure"),
            Output("metrics-graph",           "figure"),
            Output("param-heatmap",           "figure"),
            Output("param-vel-heatmap",       "figure"),
            Output("generated-sample",        "figure"),
            Output("class-sample-container",  "children"),
            Output("sample-panel",            "style"),
            Output("class-sample-panel",      "style"),
            Input("interval",                 "n_intervals"),
            Input("run-selector",             "value"),
            State("run-type",                 "data"))
        def update_graphs(_, run_value, run_type):
            empty      = go.Figure()
            show, hide = {}, {"display": "none"}

            if not run_value:
                return empty, empty, empty, empty, empty, [], show, hide

            run_dir = self.root / run_value
            try:
                loss_fig  = _build_loss_figure(run_dir)
                param_fig = _build_param_heatmap(run_dir)
                vel_fig   = _build_param_velocity_heatmap(run_dir)

                if run_type == "xmap":
                    meta        = _read_metadata(run_dir)
                    num_classes = int(meta.get("num_classes", 2))
                    metrics_fig = _build_xmap_metrics_figure(run_dir, num_classes)
                    class_cards = _build_xmap_class_samples(
                        run_dir, self.sampler, num_classes)
                    return (loss_fig, metrics_fig, param_fig, vel_fig,
                            empty, class_cards, hide, show)
                else:
                    metrics_fig = _build_metrics_figure(run_dir)
                    sample_fig  = _build_sample_plot(run_dir, self.sampler)
                    return (loss_fig, metrics_fig, param_fig, vel_fig,
                            sample_fig, [], show, hide)

            except Exception as e:
                err = go.Figure()
                err.add_annotation(text=str(e), xref="paper", yref="paper",
                                   x=0.5, y=0.5, showarrow=False)
                return err, err, err, err, err, [
                    html.P(f"Error: {e}", style={"color": "red"})], show, hide

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self, port=None):
        if port is None:
            with socket.socket() as s:
                s.bind(('', 0))
                port = s.getsockname()[1]
        ip = socket.gethostbyname(socket.gethostname())
        print(f"Scanning: {self.root}")
        print(f"Dashboard URL: http://{ip}:{port}")
        self.app.run(host="0.0.0.0", debug=False, port=port)


# Backwards-compatible aliases
TrainingDashboard    = QGANDashboard
XMapCQGANDashboard   = QGANDashboard