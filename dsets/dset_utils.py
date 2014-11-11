import numpy as np
from ops import array

def one_hot(data, n):
    data_1h = np.zeros((n, data.shape[0], data.shape[1]))
    for t in xrange(data.shape[0]):
        for b in xrange(data.shape[1]):
            data_1h[data[t, b], t, b] = 1
    return array(data_1h)
