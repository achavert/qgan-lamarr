import numpy as np
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
        