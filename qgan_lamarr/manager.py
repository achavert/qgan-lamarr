from pathlib import Path
import json
from datetime import datetime
from qiskit import qasm3
import pickle

class FileManager:
    def __init__(self, _gen, _dis, metadata):
        
        
        self.gen = _gen
        self.dis = _dis
        self.metadata = metadata
        
        self.output_dir = "./output"

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        
        self.run_dir = Path(self.output_dir) / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Monitoring run: run_{self.timestamp}")
        
        self.param_file = self.run_dir / "params.csv"
        self.loss_file = self.run_dir / "losses.csv"
        self.metrics_file = self.run_dir / "metrics.csv"
        self.metadata_file = self.run_dir / "meta.json"
        self.generator_file = self.run_dir / "generator_circuit.qasm"
        self.discriminator_file = self.run_dir / "discriminator_model.keras"

        self.metrics_header_written = False
        
        self.create_files()

    def create_files(self):
        self.param_file.touch()
        with open(self.loss_file, "w") as f:
            f.write("step,generator_loss,discriminator_loss\n")
            
        self.metrics_file.touch()
        with open(self.metadata_file, "w") as f:
            _data = {'timestamp': self.timestamp, **self.metadata}
            json.dump(_data, f, indent = 4, default = _serialize)

        with open(self.generator_file, "wb") as f:
            pickle.dump(self.gen, f)

        self.dis.save(self.discriminator_file)

    def update_param(self, _params):
        with open(self.param_file, "a") as f:
            f.write(",".join(map(str, _params)) + "\n")
            
    def update_losses(self, _step, _gen_loss, _dis_loss):
        with open(self.loss_file, "a") as f:
            f.write(f"{_step},{_gen_loss},{_dis_loss}\n")
                
    def update_metrics(self, _step, _metrics):
        if not self.metrics_header_written:
            header = "step," + ",".join(_metrics.keys()) + "\n"
            with open(self.metrics_file, "w") as f:
                f.write(header)
            self.metrics_header_written = True

        values = ",".join(map(str, _metrics.values()))

        with open(self.metrics_file, "a") as f:
            f.write(f"{_step},{values}\n")
            
    def update_distribution(self, step, sample):

        outdir = self.run_dir / "samples"
        outdir.mkdir(exist_ok=True)
    
        file = outdir / f"step_{step:05d}.json"
    
        with open(file, "w") as f:
            json.dump(sample, f)

      
def _serialize(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
    

    
        
        

