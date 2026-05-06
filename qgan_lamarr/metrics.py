import numpy as np
from .tools import dict2vector, dict2sample
from scipy.stats import kstest, wasserstein_distance, chi2_contingency
from scipy.spatial.distance import jensenshannon



def scoreKS(sample1: dict, sample2: dict) -> float:
    sample1_num = dict2sample(sample1)
    sample2_num = dict2sample(sample2)
    stat, pvalue = kstest(sample1_num, sample2_num)
    return stat

def wasserstein(sample1: dict, sample2: dict) -> float:
    sample1_num = dict2sample(sample1)
    sample2_num = dict2sample(sample2)
    stat = wasserstein_distance(sample1_num, sample2_num)
    return stat

def jensen_shannon(sample1: dict, sample2: dict, bins) -> float:
    sample1_num = dict2vector(sample1, bins)
    sample2_num = dict2vector(sample2, bins)
    stat = jensenshannon(sample1_num, sample2_num)
    return stat

def chi2(sample1: dict, sample2: dict) -> float:
    keys = sorted(set(sample1.keys()).union(sample2.keys()))
    sample1_list = np.array([sample1.get(k, 0) for k in keys], dtype = float)
    sample2_list = np.array([sample2.get(k, 0) for k in keys], dtype = float)

    table = np.vstack([sample1_list, sample2_list])
        
    chi2, p, dof, expected = chi2_contingency(table)
    return chi2
    
def kullback_leibler_divergence(sample1: dict, sample2: dict, bins) -> float:
    vec1 = dict2vector(sample1, bins) + 1e-12 # Avoid zero division
    vec2 = dict2vector(sample2, bins) + 1e-12 
        
    vec1 = vec1 / np.sum(vec1)
    vec2 = vec2 / np.sum(vec2)

    return float(np.sum(vec1 * np.log(vec1 / vec2)))

def generator_entropy(sample1: dict, bins) -> float:
    vec1 = dict2vector(sample1, bins) + 1e-12 # Avoid zero division
    vec1 = vec1 / np.sum(vec1)
    return float(-np.sum(vec1 * np.log(vec1)))

def fidelity(sample1: dict, sample2: dict, bins) -> float:
    '''
    Classical fidelity (Bhattacharyya coefficient squared)
    '''
    vec1 = dict2vector(sample1, bins)
    vec2 = dict2vector(sample2, bins)
    return float(np.sum(np.sqrt(vec1 * vec2))**2)


# def compute_baseline_js(n_samples: int = 10) -> float:
#     '''
#     Average JS divergence between two independent samples of the real distribution (sampling noise floor).
#     '''
#     baseline_values = []
#     for _ in range(n_samples):
#         sample_a = self.real_dist_eval()
#         sample_b = self.real_dist_eval()
            
#         vec_a = self.dict2vector(sample_a)
#         vec_b = self.dict2vector(sample_b)
#         baseline_values.append(jensenshannon(vec_a, vec_b))
            
#     return float(np.mean(baseline_values)), float(np.std(baseline_values)) 

def metric_avg(_epoch, _metric, avg_steps = 20):
    if _epoch < avg_steps:
        return np.mean(_metric[0:])
    else:
        return np.mean(_metric[_epoch-avg_steps:_epoch+1])