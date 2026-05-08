import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path
import csv
import json

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import io
import base64
import tempfile

import numpy as np
import pickle
from qiskit import QuantumCircuit, qasm3
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import circuit_drawer
import tensorflow as tf


class TrainingDashboard:
    PAGE_STYLE = {
    "maxWidth": "1400px",
    "margin": "auto",
    "fontFamily": "Arial, sans-serif",
    "padding": "20px"
    }
    
    SECTION_STYLE = {
        "marginBottom": "25px"
    }
    
    GRAPH_STYLE = {
        "height": "420px"
    }
    
    IMAGE_CONTAINER_STYLE = {
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center",
        "overflow": "auto",
        "padding": "10px"
    }
    
    CIRCUIT_IMG_STYLE = {
        "maxWidth": "900px",
        "width": "100%",
        "height": "auto"
    }
    
    MODEL_IMG_STYLE = {
        "maxWidth": "400px",
        "width": "100%",
        "height": "auto"
    }
    
    GRAPH_STYLE = {
    "width": "100%",
    "height": "450px",
    "marginBottom": "30px"
    }

    SAMPLE_STYLE = {
    "width": "60%",
    "height": "600px",
    "marginBottom": "30px"
    }
    
    def __init__(self, run_dir, refresh_rate=2):
        
        self.run_dir = Path(run_dir)
        self.refresh_rate = refresh_rate

        self.param_file = self.run_dir / "params.csv"
        self.loss_file = self.run_dir / "losses.csv"
        self.metrics_file = self.run_dir / "metrics.csv"
        self.metadata_file = self.run_dir / "meta.json"
        self.generator_file = self.run_dir / "generator_circuit.qasm"
        self.discriminator_file = self.run_dir / "discriminator_model.keras"

        
        self.metadata = self.read_metadata()
        self.app = dash.Dash(__name__)
        
        self.import_circuit()
        self.sampler = StatevectorSampler()
        self.setup_layout()        
        
        self.setup_callbacks()
        
        
        

    '''
    Load data functions
    '''
    
    def read_metadata(self):
        with open(self.metadata_file, "r") as f:
            return json.load(f)

    def read_parameters(self):
        params = []
        with open(self.param_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                params.append([float(x) for x in row])
        return params

    def read_losses(self):
        losses = []
        with open(self.loss_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                losses.append({"step": int(row["step"]),
                               "generator_loss": float(row["generator_loss"]),
                               "discriminator_loss": float(row["discriminator_loss"])})
        return losses

    def read_metrics(self):
        metrics = []
        with open(self.metrics_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["step"] = int(row["step"])
                for k in row:
                    if k != "step":
                        row[k] = float(row[k])
                metrics.append(row)
        return metrics
        
    def import_circuit(self):
        with open(self.generator_file, "rb") as f:
            self.qc = pickle.load(f)
        
    def sample_circuit(self, shots = 2**10):
        params = self.read_parameters()
        if not params:
            return {}
        
        qc_gen = self.qc.copy()
        qc_gen.measure_all()
        last_params = params[-1]
        param_dict = dict(zip(self.qc.parameters, last_params))
        pub = (qc_gen, param_dict)
        job = self.sampler.run([pub], shots = shots)
        counts = job.result()[0].data.meas.get_counts()
        return counts
        
    def read_training(self):
        params = self.read_parameters()
        losses = self.read_losses()
        metrics = self.read_metrics()

    

    '''
    Build model summary
    '''

    
    
    def build_circuit_figure(self):
        fig = circuit_drawer(self.qc, output = "mpl")
    
        buf = io.BytesIO()
        fig.savefig(buf, format = "png", bbox_inches = "tight")
        buf.seek(0)
    
        img_base64 = base64.b64encode(buf.read()).decode()
    
        return html.Div(html.Img(src = f"data:image/png;base64,{img_base64}", style = self.CIRCUIT_IMG_STYLE), 
                        style=self.IMAGE_CONTAINER_STYLE)

    def build_model_plot(self):
        
        model = tf.keras.models.load_model(self.discriminator_file)

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
    
            tf.keras.utils.plot_model(
                model,
                to_file=tmp.name,
                show_shapes=True,
                show_dtype=False,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                show_layer_activations=True,
                show_trainable=True
            )
    
            with open(tmp.name, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode()
    
        return html.Div(html.Img(src =  f"data:image/png;base64,{img_base64}", 
                                 style = self.MODEL_IMG_STYLE), style = self.IMAGE_CONTAINER_STYLE)
        
    def build_sample_plot(self):
        sample = self.sample_circuit()
        num_qubits = self.qc.num_qubits
        states = [format(b, f'0{int(num_qubits)}b') for b in range(2**num_qubits)]
        values = [sample.get(s,0) for s in states]

        dens_val = [v/sum(values) for v in values]
        
        fig = go.Figure()
        fig.add_bar(x = states, y= dens_val)

        fig.update_layout(title = "Generated sample",
                          xaxis_title = "Bins",
                          yaxis_title = "Counts",
                          template = "plotly_white",
                          yaxis = dict(range=[0, 1]))
        return fig

        
    '''
    Build graphs functions
    '''

    def build_metadata_table(self):
        rows = []
        for key, value in self.metadata.items():
            rows.append(html.Tr([html.Td(str(key), style={"fontWeight": "bold", "padding": "4px 8px"}),
                                 html.Td(str(value), style={"padding": "4px 8px"})]))
        return html.Table(rows, style={"borderCollapse": "collapse",
                                       "width": "100%",
                                       "marginBottom": "20px",
                                       "border": "1px solid #ccc"})
        
    def build_loss_figure(self):

        losses = self.read_losses()

        steps = [l["step"] for l in losses]
        g_loss = [l["generator_loss"] for l in losses]
        d_loss = [l["discriminator_loss"] for l in losses]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x = steps,
                                 y = g_loss,
                                 mode = "lines",
                                 name = "Generator loss"))

        fig.add_trace(go.Scatter(x = steps,
                                 y = d_loss,
                                 mode = "lines",
                                 name = "Discriminator loss"))
        if self.metadata['wasserstein']:
            ref = 0.0
            txt = '0'
        else :
            ref = np.log(2)
            txt = "-log(1/2)"

        fig.add_hline(y = ref,
                      line_dash = "dot",
                      line_color = "gray",
                      annotation_text = txt,
                      annotation_position="top right")

        fig.update_layout(title = "Losses",
                          xaxis_title = "Epoch",
                          yaxis_title = "Loss function")

        return self.standardize_figure(fig)

    def build_metrics_figure(self):

        metrics = self.read_metrics()

        if not metrics:
            return go.Figure()

        steps = [m["step"] for m in metrics]

        fig = go.Figure()

        for key in metrics[0]:
            if key != "step":
                fig.add_trace(go.Scatter(x = steps,
                                         y = [m[key] for m in metrics],
                                         mode = "lines",
                                         name = key))
                
        fig.add_hline(y = self.metadata['baseline_js'][0],
                      line_dash = "dot",
                      line_color = "black",
                      annotation_text = "baseline_js",
                      annotation_position="top left")

        fig.add_hline(y = self.metadata['baseline_js'][0] + self.metadata['baseline_js'][1],
                      line_dash = "dot",
                      line_color = "gray")

        fig.add_hline(y = self.metadata['baseline_js'][0] - self.metadata['baseline_js'][1],
                      line_dash = "dot",
                      line_color = "gray")
        
        fig.update_layout(title="Metrics",
                          xaxis_title = "Epoch")

        return self.standardize_figure(fig)

    def build_param_heatmap_figure(self):
        """
        Row = parameter, column = training step.
        """
        params = self.read_parameters()
    
        if not params:
            return go.Figure()
        
        param_array = np.array(params).T
        param_array_wrapped = np.mod(param_array, 2 * np.pi)
        # phase_colorscale = [[0.0, "yellow"],
        #                     [0.25, "green"],
        #                     [0.50, "cyan"],
        #                     [0.75, "red"],
        #                     [1.0, "yellow"]]
        phase_colorscale = [[0.0, "yellow"],
                            [0.25, "cyan"],
                            [0.75, "magenta"],
                            [1.0, "yellow"]]
        
        fig = go.Figure(data=go.Heatmap(z = param_array_wrapped,
                                        colorscale = phase_colorscale,
                                        colorbar = dict(title="Parameter value", 
                                                        tickvals=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                                                        ticktext=["0", "π/2", "π", "3π/2", "2π"]),
                                        zsmooth = False,
                                        zmin = 0,
                                        zmax = 2*np.pi))
        fig.update_yaxes(autorange="reversed")   
        fig.update_layout(title = "Parameter heatmap",
                          xaxis_title = "Epoch",
                          yaxis_title = "Parameter")
    
        return self.standardize_figure(fig)

    def build_param_velocity_heatmap_figure(self):
        '''
        Row = parameter, column = training step.
        '''
    
        params = self.read_parameters()
    
        if len(params) < 2:
            return go.Figure()
    
        param_array = np.array(params)
    
        delta = np.angle(np.exp(1j * np.diff(param_array, axis=0)))
        
        velocity = np.abs(delta).T
    
        fig = go.Figure(data=go.Heatmap(z = velocity,
                                        colorscale = "Inferno",
                                        colorbar = dict(title="|Δθ|"),
                                        zsmooth = False,
                                        zmin = 0))
        fig.update_yaxes(autorange="reversed")                   
        fig.update_layout(title = "Parameter velocity heatmap",
                          xaxis_title = "Epoch",
                          yaxis_title = "Parameter")
        return self.standardize_figure(fig)



    
    '''
    Dashboard layout
    '''

    def setup_layout(self):
        self.app.layout = html.Div([
    
            html.H1("QGAN Training Dashboard", style={"marginBottom": "30px"}),
    
            html.Details([
                html.Summary("Run metadata"),
                self.build_metadata_table()
            ], style={"marginBottom": "20px"}),
    
            html.Details([
                html.Summary("Generator circuit"),
                self.build_circuit_figure()
            ], style={"marginBottom": "20px"}),
    
            html.Details([
                html.Summary("Discriminator model"),
                self.build_model_plot()
            ], style={"marginBottom": "20px"}),
    
            dcc.Graph(id="loss-graph", style=self.GRAPH_STYLE),
    
            dcc.Graph(id="metrics-graph", style=self.GRAPH_STYLE),
    
            dcc.Graph(id="param-heatmap", style=self.GRAPH_STYLE),
    
            dcc.Graph(id="param-vel-heatmap", style=self.GRAPH_STYLE),

            dcc.Graph(id="generated-sample", style=self.SAMPLE_STYLE),
            
            dcc.Interval(
                id="interval",
                interval=self.refresh_rate * 1000,
                n_intervals=0
            )
    
        ], style={
            "maxWidth": "1400px",
            "margin": "auto",
            "padding": "20px",
            "fontFamily": "Arial"
        })
    '''
    Callbacks
    '''
    def setup_callbacks(self):
        @self.app.callback(Output("loss-graph", "figure"),
                           Output("metrics-graph", "figure"),
                           Output("param-heatmap", "figure"),
                           Output("param-vel-heatmap", "figure"),
                           Output("generated-sample", "figure"),
                           Input("interval", "n_intervals"))
        def update_graphs(_):
            loss_fig = self.build_loss_figure()
            metric_fig = self.build_metrics_figure()
            param_fig = self.build_param_heatmap_figure()
            param_vel_fig = self.build_param_velocity_heatmap_figure()
            sample_fig = self.build_sample_plot()
            return loss_fig, metric_fig, param_fig, param_vel_fig, sample_fig
                
    def standardize_figure(self, fig):
        fig.update_layout(
            template="plotly_white",
            height=450,
            margin=dict(l=60, r=30, t=50, b=50),
            xaxis=dict(title="Epoch")
        )
    
        return fig


    
    def run(self, local = True, port = None):
        if local:
            print(f"Monitoring run: {self.run_dir}")
            print(f"Monitoring host: http://localhost:{port}")
            self.app.run(debug = True, port = port)
        else :
            import socket
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            print(f"Monitoring run:  {self.run_dir}")
            print(f"Dashboard URL:   http://{ip}:{port}")
            self.app.run(host="0.0.0.0", debug=False, port=port)