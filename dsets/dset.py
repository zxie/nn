import numpy as np

# TODO Set up so only loads parts of dataset (a few files)
# at a time for large datasets

class Dataset(object):

    def __init__(self, feat_dim, batch_size):
        self.ind = 0
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        self.data = None
        self.data_ind = 0
        # Shuffle consistently
        np.random.seed(19)

    def data_left(self):
        return self.data_ind < self.data.shape[1]

    def get_batch(self):
        raise NotImplementedError()

    def restart(self, shuffle=False):
        ''' For starting a new epoch '''
        self.data_ind = 0
        if shuffle:
            np.random.shuffle(self.data.T)
