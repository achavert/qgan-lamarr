import numpy as np


'''
Distributions
'''
def SingleGaussian(mean = 0.0, sd = 1.0, shots = 2**10):
    return np.random.normal(loc = mean, scale = sd, size = shots)

def MixedGaussian(mean = [-2.0, 2.0], sd = [0.5, 0.5], shots = 2**10):
    if len(mean) != len(sd):
        raise Exception('Mean and sd array length does not match')
        
    if shots%len(mean) != 0:
        raise Exception('Number of samples non divisible by the number of mixed distributions')
        
    samp = np.array([])
    for d in range(len(mean)):
        samp = np.concatenate([samp, np.random.normal(loc = mean[d], scale = sd[d], size = shots//len(mean))])
    return samp



'''
Binning function
'''
def MinMaxBinning(_data, _nbins):
    data_interval = max(_data) - min(_data)
    bin_length = float(data_interval)/float(_nbins)
    
    binned_data = {}
    for b in range(_nbins):
        bin_min_val = min(_data) + bin_length * b
        bin_max_val = min(_data) + bin_length * (b+1)
        
        bin_counts = sum(1 for x in _data if bin_min_val <= x < bin_max_val)
        if b == _nbins-1: 
            bin_counts += sum(1 for x in _data if x == bin_max_val)

        binned_data.update({format(int(b), f'0{int(np.log2(_nbins))}b') : bin_counts})
    return binned_data

def RangeBinning(_data, _nbins, _range):
    data_interval = _range[1] - _range[0]
    bin_length = float(data_interval)/float(_nbins)
    
    binned_data = {}
    for b in range(_nbins):
        bin_min_val = _range[0] + bin_length * b
        bin_max_val = _range[0] + bin_length * (b+1)
        
        bin_counts = sum(1 for x in _data if bin_min_val <= x < bin_max_val)
        if b == _nbins-1: 
            bin_counts += sum(1 for x in _data if x == bin_max_val)

        binned_data.update({format(int(b), f'0{int(np.log2(_nbins))}b') : bin_counts})
    return binned_data