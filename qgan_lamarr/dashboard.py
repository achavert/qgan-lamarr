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
#  XMapQCGAN figure builders  (stateless)
# ══════════════════════════════════════════════════════════════════════════════

def _load_xmap(run_dir: Path) -> list:
    """Load the pickled xmap list of QuantumCircuits saved by FileManager.save_xmap."""
    with open(run_dir / "xmap.pkl", "rb") as f:
        return pickle.load(f)


def _xmap_sample_circuit(run_dir: Path, sampler: StatevectorSampler,
                         label: int, shots: int = 2**10) -> dict:
    """
    Reproduce XMapQCGAN.cond_generator_eval exactly:
    load the saved xmap circuit for this label, compose with the ansatz,
    bind the latest parameters and sample.
    Works for both the default xmap and any custom xmap passed by the user.
    """
    params = _read_parameters(run_dir)
    if not params:
        return {}
    qc     = _load_circuit(run_dir)
    xmap   = _load_xmap(run_dir)
    qc_run = xmap[label].compose(qc, range(qc.num_qubits))
    qc_run.measure_all()
    param_dict = dict(zip(qc.parameters, params[-1]))
    job = sampler.run([(qc_run, param_dict)], shots=shots)
    return job.result()[0].data.meas.get_counts()


def _build_xmap_metrics_figure(run_dir: Path, num_classes: int):
    """
    Metrics figure for XMapQCGAN.
    Per-class keys  : jensen_shannon_c{k}, fidelity_c{k}
    Aggregate keys  : jensen_shannon, jensen_shannon_avg, fidelity, fidelity_avg
    """
    met = _read_metrics(run_dir)
    if not met:
        return go.Figure()
    meta  = _read_metadata(run_dir)
    steps = [m["step"] for m in met]
    fig   = go.Figure()

    # Per-class JS — left axis, dotted
    for k in range(num_classes):
        key   = f"jensen_shannon_c{k}"
        color = _CLASS_COLORS[k % len(_CLASS_COLORS)]
        if key in met[0]:
            fig.add_trace(go.Scatter(
                x=steps, y=[m[key] for m in met],
                mode="lines", name=f"JS c{k}",
                line=dict(color=color, dash="dot"), opacity=0.55))

    # Per-class fidelity — right axis, solid
    for k in range(num_classes):
        key   = f"fidelity_c{k}"
        color = _CLASS_COLORS[k % len(_CLASS_COLORS)]
        if key in met[0]:
            fig.add_trace(go.Scatter(
                x=steps, y=[m[key] for m in met],
                mode="lines", name=f"Fidelity c{k}",
                line=dict(color=color), opacity=0.45,
                yaxis="y2"))

    # Aggregate JS — left axis, bold
    for key, label, style in [
        ("jensen_shannon",     "JS aggregate",       dict(color="black", width=2)),
        ("jensen_shannon_avg", "JS aggregate (avg)", dict(color="black", width=2, dash="dash")),
    ]:
        if key in met[0]:
            fig.add_trace(go.Scatter(
                x=steps, y=[m[key] for m in met],
                mode="lines", name=label, line=style))

    # Aggregate fidelity — right axis, bold
    for key, label, style in [
        ("fidelity",     "Fidelity aggregate",       dict(color="darkblue", width=2)),
        ("fidelity_avg", "Fidelity aggregate (avg)", dict(color="darkblue", width=2, dash="dash")),
    ]:
        if key in met[0]:
            fig.add_trace(go.Scatter(
                x=steps, y=[m[key] for m in met],
                mode="lines", name=label, line=style,
                yaxis="y2"))

    # Baseline JS band
    baseline = meta.get("baseline_js")
    if baseline:
        mean_js, std_js = baseline if isinstance(baseline, (list, tuple)) else (baseline, 0)
        fig.add_hline(y=mean_js, line_dash="dot", line_color="gray",
                      annotation_text="baseline JS", annotation_position="top left")
        fig.add_hline(y=mean_js + std_js, line_dash="dot", line_color="lightgray")
        fig.add_hline(y=max(0., mean_js - std_js), line_dash="dot", line_color="lightgray")

    fig.update_layout(
        title="Conditional metrics — XMapQCGAN",
        xaxis_title="Epoch",
        yaxis=dict(title="Jensen-Shannon divergence", range=[0, 1]),
        yaxis2=dict(title="Fidelity", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white", height=520,
        margin=dict(l=60, r=60, t=70, b=50))
    return fig


def _build_xmap_class_samples(run_dir: Path, sampler: StatevectorSampler,
                               num_classes: int) -> list:
    """
    2-column grid of bar charts, one per class.
    Each chart title shows the input basis state |bitstring>.
    """
    qc     = _load_circuit(run_dir)
    n      = qc.num_qubits
    bins   = [format(b, f"0{n}b") for b in range(2 ** n)]
    dtick  = max(1, 2**n // 16)

    cards = []
    for k in range(num_classes):
        sample = _xmap_sample_circuit(run_dir, sampler, k)
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
            dcc.Graph(figure=fig, id=f"xmap-sample-c{k}"),
            style={
                "width": "48%",
                "display": "inline-block",
                "verticalAlign": "top",
                "marginBottom": "16px",
                "marginRight": "2%" if k % 2 == 0 else "0",
            }))
    return cards



def _build_xmap_circuit_panels(run_dir: Path) -> html.Div:
    """
    Render each xmap encoding circuit as a card in a responsive grid.
    """
    import matplotlib.pyplot as plt
    qc   = _load_circuit(run_dir)
    xmap = _load_xmap(run_dir)
    n    = qc.num_qubits
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
            # Coloured header bar
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
            # Circuit image
            html.Div(
                html.Img(src=f"data:image/png;base64,{img}",
                         style={"width": "100%", "height": "120px",
                                "objectFit": "contain", "display": "block"}),
                style={"padding": "6px", "background": "white",
                       "height": "120px", "overflow": "hidden"}),
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
    """Return 'xmap' if meta.json contains num_classes, else 'qgan'."""
    try:
        meta = _read_metadata(run_dir)
        return 'xmap' if 'num_classes' in meta else 'qgan'
    except Exception:
        return 'qgan'


def _list_runs(root: Path) -> list[str]:
    """
    Recursively find all run directories (containing meta.json) under root.
    Returns a sorted list of path strings relative to root, newest first.
    """
    if not root.exists():
        return []
    return sorted(
        [str(p.parent.relative_to(root)) for p in root.rglob("meta.json")
         if p.parent.is_dir()],
        reverse=True)


def _build_tree(root: Path, all_runs: list[str], selected: str | None = None) -> html.Div:
    """
    Build a collapsible folder tree from the list of run paths.

    Each intermediate directory becomes an html.Details/Summary (unfoldable).
    Each run leaf becomes a clickable row that sets the selected run via a
    clientside callback writing to the 'selected-run' Store.

    Run type is detected from meta.json and shown as a small badge:
        QGAN  (grey)  or  CQGAN  (blue)
    """
    # Group runs by their top-level folder for tree construction
    # We build a nested dict: {folder: {subfolder: ... : [run_name]}}
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

        # Folders first (sorted), runs after
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

            # Read epoch count if available
            try:
                meta   = _read_metadata(run_dir)
                epochs = meta.get('epochs', '?')
                ts     = meta.get('timestamp', '')
                detail = f"  {epochs} epochs"
                if ts:
                    detail += f"  ·  {ts[:8]} {ts[9:15] if len(ts) > 9 else ''}"
            except Exception:
                detail = ''

            badge_color = "#4a90d9" if run_type == 'xmap' else "#888"
            badge_label = "CQGAN" if run_type == 'xmap' else "QGAN"

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
    """
    Unified dashboard for QGAN and XMapQCGAN runs.

    Left sidebar shows the full folder tree rooted at `root`, with collapsible
    folders (html.Details/Summary) and clickable run items.  Each run shows a
    type badge (QGAN / CQGAN) and epoch count read from meta.json.

    Clicking a run loads it in the main panel.  The run type is auto-detected
    and the correct graphs are shown:
        QGAN      -> metrics + single sample plot
        XMapQCGAN -> conditional metrics + per-class sample grid

    Parameters
    ----------
    root         : root directory to scan recursively.  Defaults to '.'.
    refresh_rate : live-update interval in seconds (default 2).

    Usage
    -----
        from qgan_lamarr import QGANDashboard
        QGANDashboard('.').run()
        QGANDashboard('/scratch/myproject').run()
    """

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
            label    = ('Generator ansatz' if run_type == 'xmap'
                        else 'Generator circuit')
            crumb    = f'📂 {run_value}'
            try:
                panels = [
                    html.Details([html.Summary('Run metadata'),
                                  _build_metadata_table(run_dir)],
                                 style={'marginBottom': '20px'}),
                    html.Details([html.Summary(label),
                                  _build_circuit_figure(run_dir)],
                                 style={'marginBottom': '20px'}),
                ]
                if run_type == 'xmap' and (run_dir / 'xmap.pkl').exists():
                    panels.append(
                        html.Details([html.Summary('Conditional circuits per class'),
                                      _build_xmap_circuit_panels(run_dir)],
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
                if run_type == 'xmap':
                    meta        = _read_metadata(run_dir)
                    num_classes = int(meta.get('num_classes', 2))
                    met_fig     = _build_xmap_metrics_figure(run_dir, num_classes)
                    cards       = _build_xmap_class_samples(
                                      run_dir, self.sampler, num_classes)
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