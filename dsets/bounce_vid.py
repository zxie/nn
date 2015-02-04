from dset import Dataset
from bounce import bounce_vec
import numpy as np
from ops import array

# Could generate infinite number of batches but want to start annealing
# after reasonably large pass
NUM_EXAMPLES_FACTOR = 100

# FIXME PARAM
T = 4
N = 1

class BounceVideo(Dataset):

    def __init__(self, feat_dim, batch_size):
        assert int(np.sqrt(feat_dim)) ** 2 == feat_dim
        super(BounceVideo, self).__init__(feat_dim, batch_size)
        self.batch_ind = 0

    def data_left(self):
        return self.batch_ind < self.feat_dim * NUM_EXAMPLES_FACTOR

    def get_batch(self):
        self.batch = np.zeros((self.feat_dim, T, self.batch_size))
        self.batch_labels = np.zeros((self.feat_dim, T, self.batch_size))

        for k in xrange(self.batch_size + 1):
            v = bounce_vec(int(np.sqrt(self.feat_dim)), n=N, T=T).T
            if k < self.batch_size:
                self.batch[:, :, k] = v
            if k > 0:
                self.batch_labels[:, :, k-1] = v

        self.batch_ind += 1
        return array(self.batch), array(self.batch_labels)
        # NOTE Can sanity check by training identity
        #return array(self.batch), array(self.batch)

    def restart(self, shuffle=True):
        self.batch_ind = 0
