import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path
import csv
import json
import socket

import dash
from dash import dcc, html, Input, Output, State, ALL, ctx
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
_LAYOUT          = {"display": "flex", "gap": "24px", "alignItems": "flex-start"}
_SIDEBAR         = {"width": "280px", "flexShrink": "0", "position": "sticky",
                    "top": "20px", "maxHeight": "90vh", "overflowY": "auto",
                    "background": "#f8f8f8", "border": "1px solid #e0e0e0",
                    "borderRadius": "6px", "padding": "12px"}
_MAIN            = {"flex": "1", "minWidth": "0"}
_SIDEBAR_HEADER  = {"fontSize": "12px", "fontWeight": "700", "color": "#666",
                    "textTransform": "uppercase", "letterSpacing": "0.05em",
                    "marginBottom": "8px", "paddingBottom": "6px",
                    "borderBottom": "1px solid #e0e0e0"}


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

def _load_xmap(run_dir: Path) -> list:
    with open(run_dir / "xmap.pkl", "rb") as f:
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

def _build_circuit_figure(run_dir: Path, run_type: str = 'qgan'):
    obj = _load_circuit(run_dir)
    
    if run_type == 'qcgan':
        # obj is CondGenerator1D: Reconstruct example circuit for class 0
        qc = QuantumCircuit(obj.num_qubits)
        for key in obj.schedule:
            if 'X_' in key:
                qc = qc.compose(obj.schedule[key][0])
            elif 'G_' in key:
                qc = qc.compose(obj.schedule[key])
    else:
        qc = obj

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


# ══════════════════════════════════════════════════════════════════════════════
#  Conditional figure builders (Handles both XMapQCGAN and QCGAN)
# ══════════════════════════════════════════════════════════════════════════════

def _cond_sample_circuit(run_dir: Path, sampler: StatevectorSampler,
                         label: int, run_type: str, shots: int = 2**10) -> dict:
    """
    Uniform sampler wrapper for old XMapQCGAN and new QCGAN.
    """
    params = _read_parameters(run_dir)
    if not params:
        return {}
    
    obj = _load_circuit(run_dir)
    
    if run_type == 'xmap':
        qc = obj
        xmap = _load_xmap(run_dir)
        qc_run = xmap[label].compose(qc, range(qc.num_qubits))
    elif run_type == 'qcgan':
        qc_run = QuantumCircuit(obj.num_qubits)
        for key in obj.schedule:
            if 'X_' in key:
                qc_run = qc_run.compose(obj.schedule[key][label])
            elif 'G_' in key:
                qc_run = qc_run.compose(obj.schedule[key])
    else:
        return {}

    qc_run.measure_all()
    param_dict = dict(zip(qc_run.parameters, params[-1]))
    job = sampler.run([(qc_run, param_dict)], shots=shots)
    return job.result()[0].data.meas.get_counts()


def _build_cond_metrics_figure(run_dir: Path, num_classes: int, run_type: str):
    """
    Metrics figure for Conditional GANs.
    """
    met = _read_metrics(run_dir)
    if not met:
        return go.Figure()
    meta  = _read_metadata(run_dir)
    steps = [m["step"] for m in met]
    fig   = go.Figure()

    for k in range(num_classes):
        key   = f"jensen_shannon_c{k}"
        color = _CLASS_COLORS[k % len(_CLASS_COLORS)]
        if key in met[0]:
            fig.add_trace(go.Scatter(
                x=steps, y=[m[key] for m in met],
                mode="lines", name=f"JS c{k}",
                line=dict(color=color, dash="dot"), opacity=0.55))

    for k in range(num_classes):
        key   = f"fidelity_c{k}"
        color = _CLASS_COLORS[k % len(_CLASS_COLORS)]
        if key in met[0]:
            fig.add_trace(go.Scatter(
                x=steps, y=[m[key] for m in met],
                mode="lines", name=f"Fidelity c{k}",
                line=dict(color=color), opacity=0.45,
                yaxis="y2"))

    for key, label, style in [
        ("jensen_shannon",     "JS aggregate",       dict(color="black", width=2)),
        ("jensen_shannon_avg", "JS aggregate (avg)", dict(color="black", width=2, dash="dash")),
    ]:
        if key in met[0]:
            fig.add_trace(go.Scatter(
                x=steps, y=[m[key] for m in met],
                mode="lines", name=label, line=style))

    for key, label, style in [
        ("fidelity",     "Fidelity aggregate",       dict(color="darkblue", width=2)),
        ("fidelity_avg", "Fidelity aggregate (avg)", dict(color="darkblue", width=2, dash="dash")),
    ]:
        if key in met[0]:
            fig.add_trace(go.Scatter(
                x=steps, y=[m[key] for m in met],
                mode="lines", name=label, line=style,
                yaxis="y2"))

    baseline = meta.get("baseline_js")
    if baseline:
        mean_js, std_js = baseline if isinstance(baseline, (list, tuple)) else (baseline, 0)
        fig.add_hline(y=mean_js, line_dash="dot", line_color="gray",
                      annotation_text="baseline JS", annotation_position="top left")
        fig.add_hline(y=mean_js + std_js, line_dash="dot", line_color="lightgray")
        fig.add_hline(y=max(0., mean_js - std_js), line_dash="dot", line_color="lightgray")

    title = "Conditional metrics — QCGAN" if run_type == 'qcgan' else "Conditional metrics — XMapQCGAN"
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis=dict(title="Jensen-Shannon divergence", range=[0, 1]),
        yaxis2=dict(title="Fidelity", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white", height=520,
        margin=dict(l=60, r=60, t=70, b=50))
    return fig


def _build_cond_class_samples(run_dir: Path, sampler: StatevectorSampler,
                               num_classes: int, run_type: str) -> list:
    """
    2-column grid of bar charts, one per class.
    """
    obj = _load_circuit(run_dir)
    n = obj.num_qubits
    bins  = [format(b, f"0{n}b") for b in range(2 ** n)]
    dtick = max(1, 2**n // 16)

    cards = []
    for k in range(num_classes):
        sample = _cond_sample_circuit(run_dir, sampler, k, run_type)
        values = [sample.get(s, 0) for s in bins]
        total  = sum(values) or 1
        color  = _CLASS_COLORS[k % len(_CLASS_COLORS)]

        fig = go.Figure()
        fig.add_bar(x=bins, y=[v / total for v in values], marker_color=color)
        fig.update_layout(
            title=f"Class {k}  —  input |{bins[k]}>",
            xaxis_title="Output bins", yaxis_title="Probability",
            template="plotly_white",
            yaxis=dict(range=[0, 1]),
            xaxis=dict(tickmode="linear", tick0=0, dtick=dtick),
            height=320,
            margin=dict(l=45, r=15, t=45, b=40),
            showlegend=False)

        cards.append(html.Div(
            dcc.Graph(figure=fig, id=f"cond-sample-c{k}"),
            style={
                "width": "48%",
                "display": "inline-block",
                "verticalAlign": "top",
                "marginBottom": "16px",
                "marginRight": "2%" if k % 2 == 0 else "0",
            }))
    return cards


def _build_cond_circuit_panels(run_dir: Path, run_type: str) -> html.Div:
    """
    Render conditional encoding circuits.
    """
    import matplotlib.pyplot as plt
    obj = _load_circuit(run_dir)
    
    if run_type == 'xmap':
        n = obj.num_qubits
        xmap = _load_xmap(run_dir)
    elif run_type == 'qcgan':
        n = obj.num_qubits
        xmap = None
        for key in obj.schedule:
            if 'X_' in key:
                xmap = obj.schedule[key]
                break
        if xmap is None:
            return html.Div("No conditional input layer found in schedule.", style={"padding": "10px"})
    else:
        return html.Div()

    bins = [format(b, f"0{n}b") for b in range(2 ** n)]

    cards = []
    for k, enc in enumerate(xmap):
        color = _CLASS_COLORS[k % len(_CLASS_COLORS)]

        fig = circuit_drawer(enc, output="mpl", style={"figwidth": 4})
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close("all")

        cards.append(html.Div([
            html.Div([
                html.Span(f"Class {k}",
                          style={"fontWeight": "700", "fontSize": "13px",
                                 "color": "white"}),
                html.Span(f"  |{bins[k]}>",
                          style={"fontWeight": "400", "fontSize": "12px",
                                 "color": "rgba(255,255,255,0.8)",
                                 "marginLeft": "6px", "fontFamily": "monospace"}),
            ], style={"background": color, "padding": "6px 10px",
                      "borderRadius": "6px 6px 0 0"}),
            html.Div(
                html.Img(src=f"data:image/png;base64,{img}",
                         style={"width": "100%", "height": "240px",
                                "objectFit": "contain", "display": "block"}),
                style={"padding": "6px", "background": "white",
                       "height": "240px", "overflow": "hidden"}),
        ], style={
            "display": "inline-block",
            "verticalAlign": "top",
            "width": "calc(50% - 8px)",
            "marginRight": "8px" if k % 2 == 0 else "0",
            "marginBottom": "12px",
            "borderRadius": "6px",
            "border": "1px solid #e0e0e0",
            "overflow": "hidden",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
        }))

    return html.Div(cards, style={"lineHeight": "0"})


# ══════════════════════════════════════════════════════════════════════════════
#  Run selector helper
# ══════════════════════════════════════════════════════════════════════════════

def _detect_run_type(run_dir: Path) -> str:
    """Return 'xmap' if xmap.pkl exists, 'qcgan' if num_classes but no xmap, else 'qgan'."""
    try:
        meta = _read_metadata(run_dir)
        if 'num_classes' in meta:
            if (run_dir / "xmap.pkl").exists():
                return 'xmap'
            else:
                return 'qcgan'
        return 'qgan'
    except Exception:
        return 'qgan'


def _list_runs(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        [str(p.parent.relative_to(root)) for p in root.rglob("meta.json")
         if p.parent.is_dir()],
        reverse=True)


def _build_tree(root: Path, all_runs: list[str], selected: str | None = None) -> html.Div:
    def insert(tree, parts, full_path):
        if len(parts) == 1:
            tree.setdefault('__runs__', []).append(full_path)
        else:
            tree.setdefault(parts[0], {})
            insert(tree[parts[0]], parts[1:], full_path)

    tree = {}
    for run in all_runs:
        parts = Path(run).parts
        insert(tree, parts, run)

    def render_node(node: dict, depth: int = 0) -> list:
        items = []
        indent = depth * 16

        for key in sorted(k for k in node if k != '__runs__'):
            child_runs = _count_runs(node[key])
            summary_style = {
                "cursor": "pointer",
                "padding": "5px 8px",
                "fontWeight": "600",
                "fontSize": "13px",
                "color": "#444",
                "listStyle": "none",
                "userSelect": "none",
            }
            items.append(html.Details([
                html.Summary(
                    [html.Span("📁 ", style={"marginRight": "4px"}),
                     key,
                     html.Span(f"  ({child_runs})",
                               style={"color": "#999", "fontWeight": "400",
                                      "fontSize": "11px"})],
                    style=summary_style),
                html.Div(render_node(node[key], depth + 1),
                         style={"borderLeft": "2px solid #e0e0e0",
                                "marginLeft": "12px"}),
            ], open=(depth == 0),
               style={"marginLeft": f"{indent}px", "marginBottom": "2px"}))

        for run_path in sorted(node.get('__runs__', []), reverse=True):
            run_dir  = root / run_path
            run_type = _detect_run_type(run_dir)
            run_name = Path(run_path).name

            try:
                meta   = _read_metadata(run_dir)
                epochs = meta.get('epochs', '?')
                ts     = meta.get('timestamp', '')
                detail = f"  {epochs} epochs"
                if ts:
                    detail += f"  ·  {ts[:8]} {ts[9:15] if len(ts) > 9 else ''}"
            except Exception:
                detail = ''

            badge_color = "#4a90d9" if run_type in ['xmap', 'qcgan'] else "#888"
            badge_label = "CQGAN" if run_type in ['xmap', 'qcgan'] else "QGAN"

            is_selected = (run_path == selected)
            item_style = {
                "marginLeft": f"{indent + 16}px",
                "padding": "4px 8px",
                "cursor": "pointer",
                "borderRadius": "4px",
                "marginBottom": "2px",
                "transition": "background 0.15s",
                "background": "#e8f0fe" if is_selected else "transparent",
                "borderLeft": "3px solid #4a90d9" if is_selected else "3px solid transparent",
            }
            items.append(html.Div(
                id={"type": "run-item", "index": run_path},
                children=[
                    html.Span("▶ ", style={"color": "#4a90d9" if is_selected else "#aaa",
                                           "fontSize": "10px", "marginRight": "4px"}),
                    html.Span(run_name, style={"fontWeight": "600" if is_selected else "500",
                                               "fontSize": "13px"}),
                    html.Span(badge_label, style={
                        "background": badge_color, "color": "white",
                        "borderRadius": "3px", "padding": "1px 5px",
                        "fontSize": "10px", "marginLeft": "7px",
                        "verticalAlign": "middle",
                    }),
                    html.Span(detail, style={"color": "#999", "fontSize": "11px",
                                             "marginLeft": "6px"}),
                ],
                n_clicks=0,
                style=item_style,
            ))
        return items

    def _count_runs(node):
        return (len(node.get('__runs__', [])) +
                sum(_count_runs(v) for k, v in node.items() if k != '__runs__'))

    children = render_node(tree)
    if not children:
        children = [html.P("No runs found.", style={"color": "#aaa",
                                                     "fontSize": "13px",
                                                     "padding": "8px"})]
    return html.Div(children, id="run-tree")


# ══════════════════════════════════════════════════════════════════════════════
#  QGANDashboard  —  unified dashboard with folder-tree sidebar
# ══════════════════════════════════════════════════════════════════════════════

class QGANDashboard:
    def __init__(self, root: str = '.', refresh_rate: float = 2):
        self.root         = Path(root)
        self.refresh_rate = refresh_rate
        self.sampler      = StatevectorSampler()
        self.app          = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    # ── Sidebar ────────────────────────────────────────────────────────────────

    def _sidebar(self):
        return html.Div([
            html.Div('Runs', style=_SIDEBAR_HEADER),
            html.Button('↻ Refresh', id='refresh-btn',
                        style={'width': '100%', 'marginBottom': '10px',
                               'cursor': 'pointer', 'fontSize': '12px',
                               'padding': '4px'}),
            html.Div(id='run-tree-container',
                     children=_build_tree(self.root, _list_runs(self.root))),
        ], style=_SIDEBAR)

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _setup_layout(self):
        self.app.layout = html.Div([
            html.H1('QGAN Dashboard',
                    style={'marginBottom': '20px', 'fontFamily': 'Arial'}),

            dcc.Store(id='selected-run', data=None),
            dcc.Store(id='run-type',     data='qgan'),

            html.Div([
                self._sidebar(),
                html.Div([
                    html.Div(id='run-breadcrumb',
                             style={'fontFamily': 'Arial', 'fontSize': '13px',
                                    'color': '#666', 'marginBottom': '16px',
                                    'minHeight': '20px'}),
                    html.Div(id='static-panels'),
                    dcc.Graph(id='loss-graph',        style=_GRAPH),
                    dcc.Graph(id='metrics-graph',     style={**_GRAPH, 'height': '520px'}),
                    dcc.Graph(id='param-heatmap',     style=_GRAPH),
                    dcc.Graph(id='param-vel-heatmap', style=_GRAPH),
                    html.Div(id='sample-panel',
                             children=[dcc.Graph(id='generated-sample', style=_SAMPLE)]),
                    html.Div(id='class-sample-panel', children=[
                        html.H3('Generated samples per class',
                                style={'marginTop': '10px', 'marginBottom': '12px',
                                       'fontFamily': 'Arial'}),
                        html.Div(id='class-sample-container'),
                    ]),
                ], style=_MAIN),
            ], style=_LAYOUT),

            dcc.Interval(id='interval',
                         interval=int(self.refresh_rate * 1000),
                         n_intervals=0),
        ], style=_PAGE)

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _setup_callbacks(self):
        app = self.app

        @app.callback(
            Output('run-tree-container', 'children'),
            Input('refresh-btn', 'n_clicks'),
            prevent_initial_call=True)
        def refresh_tree(_):
            return _build_tree(self.root, _list_runs(self.root))

        @app.callback(
            Output('selected-run', 'data'),
            Output('run-tree-container', 'children', allow_duplicate=True),
            Input({'type': 'run-item', 'index': ALL}, 'n_clicks'),
            State('selected-run', 'data'),
            prevent_initial_call=True)
        def select_run(n_clicks_list, current):
            if not any(n_clicks_list):
                return current, dash.no_update
            triggered = ctx.triggered_id
            if triggered is None:
                return current, dash.no_update
            run_path = triggered['index']
            tree = _build_tree(self.root, _list_runs(self.root), selected=run_path)
            return run_path, tree

        @app.callback(
            Output('static-panels',  'children'),
            Output('run-type',       'data'),
            Output('run-breadcrumb', 'children'),
            Input('selected-run',    'data'))
        def update_static(run_value):
            if not run_value:
                return (html.P('Select a run from the sidebar.',
                               style={'color': '#aaa', 'fontFamily': 'Arial'}),
                        'qgan', '')
            run_dir  = self.root / run_value
            run_type = _detect_run_type(run_dir)
            label    = ('Generator ansatz' if run_type in ['xmap', 'qcgan']
                        else 'Generator circuit')
            crumb    = f'📂 {run_value}'
            try:
                if run_type == 'qcgan':
                    label += ' (Class 0 example)'

                panels = [
                    html.Details([html.Summary('Run metadata'),
                                  _build_metadata_table(run_dir)],
                                 style={'marginBottom': '20px'}),
                    html.Details([html.Summary(label),
                                  _build_circuit_figure(run_dir, run_type)],
                                 style={'marginBottom': '20px'}),
                ]
                if run_type in ['xmap', 'qcgan']:
                    panels.append(
                        html.Details([html.Summary('Conditional circuits per class'),
                                      _build_cond_circuit_panels(run_dir, run_type)],
                                     style={'marginBottom': '20px'}))
                panels.append(
                    html.Details([html.Summary('Discriminator model'),
                                  _build_model_plot(run_dir)],
                                 style={'marginBottom': '20px'}))
            except Exception as e:
                panels = html.P(f'Error: {e}', style={'color': 'red'})
            return panels, run_type, crumb

        @app.callback(
            Output('loss-graph',             'figure'),
            Output('metrics-graph',          'figure'),
            Output('param-heatmap',          'figure'),
            Output('param-vel-heatmap',      'figure'),
            Output('generated-sample',       'figure'),
            Output('class-sample-container', 'children'),
            Output('sample-panel',           'style'),
            Output('class-sample-panel',     'style'),
            Input('interval',                'n_intervals'),
            Input('selected-run',            'data'),
            State('run-type',                'data'))
        def update_graphs(_, run_value, run_type):
            empty      = go.Figure()
            show, hide = {}, {'display': 'none'}
            if not run_value:
                return empty, empty, empty, empty, empty, [], hide, hide
            run_dir = self.root / run_value
            try:
                loss_fig  = _build_loss_figure(run_dir)
                param_fig = _build_param_heatmap(run_dir)
                vel_fig   = _build_param_velocity_heatmap(run_dir)
                
                if run_type in ['xmap', 'qcgan']:
                    meta        = _read_metadata(run_dir)
                    num_classes = int(meta.get('num_classes', 2))
                    met_fig     = _build_cond_metrics_figure(run_dir, num_classes, run_type)
                    cards       = _build_cond_class_samples(
                                      run_dir, self.sampler, num_classes, run_type)
                    return (loss_fig, met_fig, param_fig, vel_fig,
                            empty, cards, hide, show)
                else:
                    met_fig    = _build_metrics_figure(run_dir)
                    sample_fig = _build_sample_plot(run_dir, self.sampler)
                    return (loss_fig, met_fig, param_fig, vel_fig,
                            sample_fig, [], show, hide)
            except Exception as e:
                err = go.Figure()
                err.add_annotation(text=str(e), xref='paper', yref='paper',
                                   x=0.5, y=0.5, showarrow=False)
                return (err, err, err, err, err,
                        [html.P(f'Error: {e}', style={'color': 'red'})],
                        show, hide)

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self, port=None):
        if port is None:
            with socket.socket() as s:
                s.bind(('', 0))
                port = s.getsockname()[1]
        ip = socket.gethostbyname(socket.gethostname())
        print(f'Scanning:      {self.root}')
        print(f'Dashboard URL: http://{ip}:{port}')
        self.app.run(host='0.0.0.0', debug=False, port=port)


# Backwards-compatible aliases
TrainingDashboard  = QGANDashboard
XMapCQGANDashboard = QGANDashboard


# ══════════════════════════════════════════════════════════════════════════════
#  EvaluationDashboard  —  static post-training evaluation view
#  Designed for publication-quality figures (LaTeX-ready exports).
# ══════════════════════════════════════════════════════════════════════════════

# ── Matplotlib / LaTeX style helpers ─────────────────────────────────────────

def _mpl_style() -> dict:
    """
    Returns a dict of rcParams that produce clean, serif figures suitable for
    direct inclusion in LaTeX documents.  Apply with `plt.rcParams.update(...)`.
    """
    return {
        "font.family":        "serif",
        "font.size":          11,
        "axes.labelsize":     12,
        "axes.titlesize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.linewidth":     0.8,
        "grid.linewidth":     0.5,
        "lines.linewidth":    1.4,
        "axes.grid":          True,
        "grid.alpha":         0.35,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    }

def _fig_to_b64(fig) -> str:
    """Serialise a matplotlib figure to a PNG base-64 string and close it."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data

def _img_panel(b64: str, caption: str = "") -> html.Div:
    """Wrap a base-64 PNG in a centred panel with an optional caption."""
    children = [html.Img(src=f"data:image/png;base64,{b64}",
                         style={"maxWidth": "820px", "width": "100%",
                                "height": "auto", "display": "block",
                                "margin": "0 auto"})]
    if caption:
        children.append(html.P(caption, style={"textAlign": "center",
                                               "color": "#555", "fontSize": "12px",
                                               "marginTop": "6px"}))
    return html.Div(children, style={"marginBottom": "32px"})


# ── Static matplotlib figure builders ────────────────────────────────────────

def _eval_loss_figure(run_dir: Path) -> str:
    """
    Training loss curves — publication style.
    Returns a base-64 PNG string.
    """
    import matplotlib.pyplot as plt

    losses = _read_losses(run_dir)
    meta   = _read_metadata(run_dir)
    steps  = [l["step"] for l in losses]
    g_loss = [l["generator_loss"]     for l in losses]
    d_loss = [l["discriminator_loss"] for l in losses]

    plt.rcParams.update(_mpl_style())
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.plot(steps, g_loss, label="Generator",     color="#2166ac")
    ax.plot(steps, d_loss, label="Discriminator", color="#d6604d")

    ref = 0.0 if meta.get("wasserstein") else np.log(2)
    lbl = "0" if meta.get("wasserstein") else r"$-\ln\!\frac{1}{2}$"
    ax.axhline(ref, ls=":", color="#555", lw=1, label=lbl)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training losses")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _eval_metrics_figure(run_dir: Path) -> str:
    """
    Training metric curves — publication style.
    Handles both QGAN (single JS/Fidelity columns) and conditional variants
    (per-class columns plus aggregates).
    Returns a base-64 PNG string.
    """
    import matplotlib.pyplot as plt

    met  = _read_metrics(run_dir)
    meta = _read_metadata(run_dir)
    if not met:
        return ""

    steps       = [m["step"] for m in met]
    is_cond     = "jensen_shannon_c0" in met[0]
    num_classes = int(meta.get("num_classes", 1)) if is_cond else 0

    plt.rcParams.update(_mpl_style())

    if is_cond:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
        cmap = plt.get_cmap("tab10")

        for k in range(num_classes):
            c = cmap(k / max(num_classes, 1))
            js_key  = f"jensen_shannon_c{k}"
            fid_key = f"fidelity_c{k}"
            if js_key in met[0]:
                ax1.plot(steps, [m[js_key]  for m in met], ls="--", alpha=0.5,
                         color=c, label=f"c{k}")
            if fid_key in met[0]:
                ax2.plot(steps, [m[fid_key] for m in met], ls="--", alpha=0.5,
                         color=c, label=f"c{k}")

        for key, ax, lbl in [
            ("jensen_shannon",     ax1, "JS aggregate"),
            ("jensen_shannon_avg", ax1, "JS avg"),
            ("fidelity",           ax2, "Fid. aggregate"),
            ("fidelity_avg",       ax2, "Fid. avg"),
        ]:
            if key in met[0]:
                ls = "-" if "avg" not in key else (0, (5, 2))
                ax.plot(steps, [m[key] for m in met], color="black",
                        lw=1.8, ls=ls, label=lbl)

        baseline = meta.get("baseline_js")
        if baseline:
            mean_js, std_js = baseline if isinstance(baseline, (list, tuple)) else (baseline, 0)
            for ax in (ax1, ax2):
                ax.axhline(mean_js, ls=":", color="#888", lw=1, label="baseline JS")
                if std_js:
                    ax.axhspan(max(0, mean_js - std_js), mean_js + std_js,
                               alpha=0.08, color="#888")

        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Jensen–Shannon divergence")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Fidelity")
        ax1.set_title("JS divergence"); ax2.set_title("Fidelity")
        ax1.set_ylim(0, 1);    ax2.set_ylim(0, 1)
        ax1.legend(frameon=False, fontsize=8, ncol=2)
        ax2.legend(frameon=False, fontsize=8, ncol=2)
    else:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for key in met[0]:
            if key != "step":
                ax.plot(steps, [m[key] for m in met], label=key)
        baseline = meta.get("baseline_js")
        if baseline:
            mean_js, std_js = baseline if isinstance(baseline, (list, tuple)) else (baseline, 0)
            ax.axhline(mean_js, ls=":", color="#888", lw=1, label="baseline JS")
            if std_js:
                ax.axhspan(max(0, mean_js - std_js), mean_js + std_js,
                           alpha=0.08, color="#888")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric value")
        ax.set_title("Training metrics")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _eval_param_figure(run_dir: Path) -> str:
    """
    Parameter evolution heatmap — publication style.
    Returns a base-64 PNG string.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    params = _read_parameters(run_dir)
    if not params:
        return ""

    z = np.mod(np.array(params).T, 2 * np.pi)   # shape (n_params, n_steps)

    plt.rcParams.update(_mpl_style())
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, z.shape[0] * 0.18 + 1.0)))

    cmap = plt.get_cmap("hsv")
    im   = ax.imshow(z, aspect="auto", origin="upper",
                     cmap=cmap, vmin=0, vmax=2 * np.pi,
                     interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\theta \ (mod \ 2\pi)$")
    cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    cbar.set_ticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameter index")
    ax.set_title("Parameter evolution")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _eval_histogram_figure(results: dict, meta: dict) -> str:
    """
    Static Real vs Generated histogram with uncertainty bands, one subplot
    per class.  Designed for direct LaTeX inclusion.
    Returns a base-64 PNG string.
    """
    import matplotlib.pyplot as plt

    is_cond     = meta.get("num_classes") is not None
    num_classes = int(meta.get("num_classes", 1))
    num_qubits  = int(np.log2(meta["bins"]))
    bins_str    = [format(b, f"0{num_qubits}b") for b in range(meta["bins"])]
    x           = np.arange(meta["bins"])
    width       = 0.38

    plt.rcParams.update(_mpl_style())
    fig, axes = plt.subplots(1, num_classes,
                             figsize=(5.0 * num_classes, 3.8),
                             squeeze=False)
    axes = axes.flatten()

    for c in range(num_classes):
        ax = axes[c]

        # Retrieve per-class mean/std vectors stored by evaluate_model
        rv = np.array(results["_real_vectors"][c])    # (n_reps, n_bins)
        mv = np.array(results["_model_vectors"][c])

        r_mean, r_std = rv.mean(axis=0), rv.std(axis=0)
        m_mean, m_std = mv.mean(axis=0), mv.std(axis=0)

        ax.bar(x - width / 2, r_mean, width, color="#888888", alpha=0.75,
               label="Real",      zorder=2)
        ax.errorbar(x - width / 2, r_mean, yerr=r_std, fmt="none",
                    ecolor="#444", elinewidth=0.9, capsize=2, zorder=3)

        ax.bar(x + width / 2, m_mean, width, color="#2166ac", alpha=0.80,
               label="Generated", zorder=2)
        ax.errorbar(x + width / 2, m_mean, yerr=m_std, fmt="none",
                    ecolor="#08306b", elinewidth=0.9, capsize=2, zorder=3)

        title = (f"Class {c}" if is_cond and num_classes > 1
                 else "Generated vs Real")
        ax.set_title(title)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Probability")
        ax.set_xticks(x)
        ax.set_xticklabels(bins_str, rotation=90 if num_qubits > 3 else 45,
                           fontsize=max(6, 10 - num_qubits))
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _run_evaluate_model(run_dir: Path, real_dist, n_reps: int = 20) -> dict:
    """
    Thin wrapper around tools.evaluate_model that also retains the per-class
    raw probability vectors needed for the histogram figure.
    """
    from .tools import dict2vector
    from .metrics import jensen_shannon, fidelity

    meta = _read_metadata(run_dir)
    shots      = int(meta.get("shots", 1024))
    is_cond    = "num_classes" in meta
    num_classes = int(meta.get("num_classes", 1)) if is_cond else 1
    num_qubits = int(np.log2(meta["bins"]))
    bins_str   = [format(b, f"0{num_qubits}b") for b in range(meta["bins"])]

    run_type = _detect_run_type(run_dir)

    params = _read_parameters(run_dir)
    if not params:
        raise RuntimeError("No parameter history found — cannot evaluate model.")
    final_weights = params[-1]

    circuit_obj = _load_circuit(run_dir)
    xmap        = _load_xmap(run_dir) if run_type == "xmap" else None
    sampler     = StatevectorSampler()

    def sample_model(c=0):
        if run_type == "qgan":
            qc = circuit_obj.copy()
        elif run_type == "xmap":
            qc = xmap[c].compose(circuit_obj, range(num_qubits))
        elif run_type == "qcgan":
            qc = QuantumCircuit(num_qubits)
            for key in circuit_obj.schedule:
                if "X_" in key:
                    qc = qc.compose(circuit_obj.schedule[key][c])
                elif "G_" in key:
                    qc = qc.compose(circuit_obj.schedule[key])
        qc.measure_all()
        param_dict = dict(zip(qc.parameters, final_weights))
        job = sampler.run([(qc, param_dict)], shots=shots)
        return job.result()[0].data.meas.get_counts()

    def sample_real(c=0):
        if is_cond:
            return real_dist(c, shots, meta["bins"])
        else:
            return real_dist(shots, meta["bins"])

    baseline_js_list,  baseline_fid_list  = [], []
    model_js_list,     model_fid_list     = [], []
    real_vectors  = {c: [] for c in range(num_classes)}
    model_vectors = {c: [] for c in range(num_classes)}

    for _ in range(n_reps):
        rep_b_js = rep_b_fid = rep_m_js = rep_m_fid = 0.0
        for c in range(num_classes):
            real_1  = sample_real(c)
            real_2  = sample_real(c)
            model_1 = sample_model(c)

            real_vectors[c].append(dict2vector(real_1, bins_str))
            model_vectors[c].append(dict2vector(model_1, bins_str))

            w = 1.0 / num_classes
            rep_b_js  += jensen_shannon(real_1, real_2,  bins_str) * w
            rep_b_fid += fidelity(real_1,       real_2,  bins_str) * w
            rep_m_js  += jensen_shannon(real_1, model_1, bins_str) * w
            rep_m_fid += fidelity(real_1,       model_1, bins_str) * w

        baseline_js_list.append(rep_b_js);   baseline_fid_list.append(rep_b_fid)
        model_js_list.append(rep_m_js);      model_fid_list.append(rep_m_fid)

    return {
        "metadata": meta,
        "baseline": {
            "jensen_shannon": {"mean": float(np.mean(baseline_js_list)),
                               "std":  float(np.std(baseline_js_list))},
            "fidelity":       {"mean": float(np.mean(baseline_fid_list)),
                               "std":  float(np.std(baseline_fid_list))},
        },
        "model": {
            "jensen_shannon": {"mean": float(np.mean(model_js_list)),
                               "std":  float(np.std(model_js_list))},
            "fidelity":       {"mean": float(np.mean(model_fid_list)),
                               "std":  float(np.std(model_fid_list))},
        },
        # Raw per-class vectors kept for the histogram figure
        "_real_vectors":  {c: real_vectors[c]  for c in real_vectors},
        "_model_vectors": {c: model_vectors[c] for c in model_vectors},
    }


def _build_eval_summary_table(results: dict) -> html.Table:
    """Render a compact HTML table summarising the evaluation metrics."""
    def _fmt(d): return f"{d['mean']:.4f} ± {d['std']:.4f}"
    rows = [
        html.Tr([html.Th("Metric", style={"padding": "6px 12px", "textAlign": "left"}),
                 html.Th("Baseline (real vs real)", style={"padding": "6px 12px"}),
                 html.Th("Model (real vs gen.)",    style={"padding": "6px 12px"})],
                style={"background": "#f0f0f0"}),
        html.Tr([html.Td("Jensen–Shannon", style={"padding": "6px 12px", "fontWeight": "600"}),
                 html.Td(_fmt(results["baseline"]["jensen_shannon"]), style={"padding": "6px 12px", "textAlign": "center"}),
                 html.Td(_fmt(results["model"]["jensen_shannon"]),    style={"padding": "6px 12px", "textAlign": "center"})]),
        html.Tr([html.Td("Fidelity",       style={"padding": "6px 12px", "fontWeight": "600"}),
                 html.Td(_fmt(results["baseline"]["fidelity"]),       style={"padding": "6px 12px", "textAlign": "center"}),
                 html.Td(_fmt(results["model"]["fidelity"]),          style={"padding": "6px 12px", "textAlign": "center"})],
                style={"background": "#fafafa"}),
    ]
    return html.Table(rows, style={
        "borderCollapse": "collapse", "width": "100%",
        "border": "1px solid #ccc", "marginBottom": "28px",
        "fontSize": "13px", "fontFamily": "Arial"})


# ── EvaluationDashboard class ─────────────────────────────────────────────────

class EvaluationDashboard:
    """
    Static, single-run evaluation dashboard.

    Shows every panel of QGANDashboard (metadata, circuit, discriminator,
    training curves, parameter heatmap) **plus** evaluation results from
    ``tools.evaluate_model``, all rendered as publication-quality matplotlib
    figures suitable for direct inclusion in LaTeX.

    There are no live-update intervals — all figures are computed once at
    page load (or on demand via the Evaluate button).

    Parameters
    ----------
    run_dir : str
        Path to the run directory produced by ``FileManager``
        (e.g. ``"./output/run_20260101_120000"``).
    real_dist : Callable
        The real-distribution sampler used during training.  Signature must
        match the one expected by ``evaluate_model``:
        ``real_dist(shots, bins) -> dict`` for a plain QGAN, or
        ``real_dist(class_idx, shots, bins) -> dict`` for a conditional model.
    n_reps : int
        Number of repetitions used to estimate evaluation uncertainties.
        Defaults to 20.
    """

    def __init__(self, run_dir: str, real_dist, n_reps: int = 20):
        self.run_dir   = Path(run_dir)
        self.real_dist = real_dist
        self.n_reps    = n_reps
        self.app       = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _setup_layout(self):
        meta     = _read_metadata(self.run_dir)
        run_type = _detect_run_type(self.run_dir)

        # ── Section helper ──
        def _section(title, children, open_=True):
            return html.Details(
                [html.Summary(title,
                              style={"fontWeight": "700", "fontSize": "15px",
                                     "cursor": "pointer", "padding": "6px 0",
                                     "userSelect": "none"}),
                 html.Div(children, style={"paddingTop": "12px"})],
                open=open_,
                style={"marginBottom": "28px",
                       "borderBottom": "1px solid #e0e0e0",
                       "paddingBottom": "8px"})

        # ── Static panels built at page-load ──
        static_panels = [
            _section("Run metadata", _build_metadata_table(self.run_dir)),
            _section("Generator circuit",
                     _build_circuit_figure(self.run_dir, run_type),
                     open_=False),
            _section("Discriminator model",
                     _build_model_plot(self.run_dir),
                     open_=False),
        ]
        if run_type in ["xmap", "qcgan"]:
            static_panels.append(
                _section("Conditional circuits per class",
                         _build_cond_circuit_panels(self.run_dir, run_type),
                         open_=False))

        # ── Training curves (matplotlib → base64) ──
        loss_b64    = _eval_loss_figure(self.run_dir)
        metrics_b64 = _eval_metrics_figure(self.run_dir)
        param_b64   = _eval_param_figure(self.run_dir)

        training_panels = [
            _section("Loss curves",
                     _img_panel(loss_b64,    "Fig. — Training losses")),
            _section("Metric curves",
                     _img_panel(metrics_b64, "Fig. — Training metrics")),
            _section("Parameter evolution",
                     _img_panel(param_b64,   "Fig. — Parameter heatmap (θ mod 2π)")),
        ]

        # ── Evaluate button + results placeholder ──
        eval_panel = html.Div([
            html.H2("Evaluation", style={"marginBottom": "12px"}),
            html.P(
                f"Shots from metadata: {meta.get('shots', 1024)}   |   "
                f"Repetitions: {self.n_reps}",
                style={"color": "#555", "fontSize": "13px", "marginBottom": "16px"}),
            html.Button(
                "▶  Run evaluation",
                id="eval-btn",
                n_clicks=0,
                style={"padding": "8px 20px", "fontSize": "14px",
                       "cursor": "pointer", "background": "#2166ac",
                       "color": "white", "border": "none",
                       "borderRadius": "5px", "marginBottom": "24px"}),
            html.Div(id="eval-status",
                     style={"color": "#888", "fontSize": "13px",
                            "marginBottom": "16px"}),
            html.Div(id="eval-results"),
        ], style={"marginBottom": "40px"})

        self.app.layout = html.Div([
            html.H1("QGAN Evaluation Dashboard",
                    style={"marginBottom": "8px", "fontFamily": "Arial"}),
            html.P(str(self.run_dir),
                   style={"color": "#777", "fontSize": "13px",
                          "marginBottom": "28px", "fontFamily": "monospace"}),

            *static_panels,
            *training_panels,
            eval_panel,

        ], style={**_PAGE, "maxWidth": "960px"})

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _setup_callbacks(self):
        app = self.app

        @app.callback(
            Output("eval-results", "children"),
            Output("eval-status",  "children"),
            Input("eval-btn",      "n_clicks"),
            prevent_initial_call=True)
        def run_evaluation(n_clicks):
            if not n_clicks:
                return dash.no_update, dash.no_update

            try:
                results = _run_evaluate_model(
                    self.run_dir, self.real_dist, n_reps=self.n_reps)
                meta    = results["metadata"]

                hist_b64 = _eval_histogram_figure(results, meta)

                summary = html.Div([
                    html.H3("Summary metrics",
                            style={"marginBottom": "10px"}),
                    _build_eval_summary_table(results),

                    html.H3("Distribution comparison",
                            style={"marginBottom": "10px"}),
                    _img_panel(hist_b64,
                               "Fig. — Real (grey) vs Generated (blue) probability "
                               "distributions with ±1σ uncertainties."),
                ])
                status = (f"✓ Evaluation complete — "
                          f"{self.n_reps} reps × {meta.get('shots', '?')} shots.")
                return summary, status

            except Exception as exc:
                return (html.P(f"Error during evaluation: {exc}",
                               style={"color": "red"}),
                        "✗ Evaluation failed.")

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, port: int | None = None):
        if port is None:
            with socket.socket() as s:
                s.bind(("", 0))
                port = s.getsockname()[1]
        ip = socket.gethostbyname(socket.gethostname())
        print(f"Run directory:    {self.run_dir}")
        print(f"Dashboard URL:    http://{ip}:{port}")
        self.app.run(host="0.0.0.0", debug=False, port=port)