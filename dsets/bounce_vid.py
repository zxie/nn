from dset import Dataset
from bounce import bounce_vec
import numpy as np
from ops import array

# Could generate infinite number of batches but want to start annealing
# after reasonably large pass
NUM_EXAMPLES_FACTOR = 10

# FIXME PARAM
N = 2

class BounceVideo(Dataset):

    def __init__(self, feat_dim, batch_size, T=20):
        assert int(np.sqrt(feat_dim)) ** 2 == feat_dim
        super(BounceVideo, self).__init__(feat_dim, batch_size)
        self.batch_ind = 0
        self.T = T

    def data_left(self):
        return self.batch_ind < self.feat_dim * NUM_EXAMPLES_FACTOR

    def get_batch(self):
        self.batch = np.zeros((self.feat_dim, self.T, self.batch_size))
        self.batch_labels = np.zeros((self.feat_dim, self.T, self.batch_size))

        for k in xrange(self.batch_size):
            v = bounce_vec(int(np.sqrt(self.feat_dim)), n=N, T=self.T+1).T
            if k < self.batch_size:
                self.batch[:, 0:self.T, k] = v[:, 0:-1]
            if k > 0:
                self.batch_labels[:, 0:self.T, k] = v[:, 1:]

        self.batch_ind += 1
        return array(self.batch), array(self.batch_labels)
        # NOTE Can sanity check by training identity
        #return array(self.batch), array(self.batch)

    def restart(self, shuffle=True):
        self.batch_ind = 0
