import numpy as np
from ops import array

def one_hot(data, n):
    data_1h = np.zeros((n, data.shape[0], data.shape[1]))
    for t in xrange(data.shape[0]):
        for b in xrange(data.shape[1]):
            data_1h[data[t, b], t, b] = 1
    return array(data_1h)

def one_hot_lists(data, n):
    T = max([len(l) for l in data])
    data_1h = np.zeros((n, T, len(data)))
    for b in xrange(len(data)):
        for t in xrange(len(data[b])):
            data_1h[data[b][t], t, b] = 1
    return array(data_1h)
