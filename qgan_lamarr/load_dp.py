import numpy as np
import pandas as pd 
import pyarrow
import dask.dataframe as ddf
import uproot
from os import environ
from glob import glob


def load_dp(mc_p_range, part, samples = 1_000_000):
    file_pattern = "/home/fredo/qgan-lamarr/LamarrTraining.root" 
    default_file_list = glob(file_pattern)
    file_list = environ.get("INPUT_FILES", " ".join(default_file_list)).split(" ")
    n_partitions = len(file_list)
    
    print (f"Found {n_partitions} files")  
    print(file_list)
    
    full_dataset = []
    for filename in file_list:
        with uproot.open(filename) as f:
            full_dataset.append(pd.DataFrame(f["TrackingTupler/reco"].arrays(library='np')))
    
    full_dataset = pd.concat(full_dataset)
    reco = (ddf.from_pandas(full_dataset).query("(type == 3) or (type == 4) or (type == 5)"))
    del full_dataset

    reco['mc_is_e'] = abs(reco.PID).isin([11]).astype(np.float32)
    reco['mc_is_mu'] = abs(reco.PID).isin([13]).astype(np.float32)
    reco['mc_is_h'] = abs(reco.PID).isin([211, 321, 2212]).astype(np.float32)
    reco['dp'] = reco.reco_p - reco.mc_p
    reco = reco.map_partitions(lambda df: df.assign(nDoF_f = df.nDoF + np.random.uniform(0, 1, len(df))))

    res_real_conditions = ['mc_p']
    res_flags = ['mc_is_e', 'mc_is_mu', 'mc_is_h']
    res_conditions = res_real_conditions + res_flags
    res_target = ['dp']
    
    df = reco[res_conditions + res_target].head(samples, npartitions = n_partitions)
    pdf = df.query(f'mc_is_{part}==1')
    pdf_range = pdf.query(f'mc_p>={mc_p_range[0]} & mc_p<={mc_p_range[1]}')
    print('Data loaded')
    return pdf_range['dp']
    
