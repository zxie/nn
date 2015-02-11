from dset import Dataset
from bounce import bounce_n
import numpy as np
from ops import array

# Could generate infinite number of batches but want to start annealing
# after reasonably large pass
NUM_EXAMPLES_FACTOR = 2000

# FIXME PARAM
N = 1

class BounceCoords(Dataset):

    def __init__(self, batch_size, T=20):
        super(BounceCoords, self).__init__(2, batch_size)
        self.batch_ind = 0
        self.T = T
        assert N == 1  # FIXME

    def data_left(self):
        return self.batch_ind < self.feat_dim * NUM_EXAMPLES_FACTOR

    def get_batch(self):
        self.batch = np.zeros((self.feat_dim, self.T, self.batch_size))
        self.batch_labels = np.zeros((self.feat_dim, self.T, self.batch_size))

        for k in xrange(self.batch_size):
            v = np.squeeze(bounce_n(T=self.T+1, n=N))
            v = v.transpose((1, 0))
            self.batch[:, 0:self.T, k] = v[:, 0:-1]
            self.batch_labels[:, 0:self.T, k] = v[:, 1:]

        self.batch_ind += 1
        return array(self.batch), array(self.batch_labels)

    def restart(self, shuffle=True):
        self.batch_ind = 0
