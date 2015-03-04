from dset import Dataset
from bounce import bounce_vec
import numpy as np
from ops import array
from bounce_vid import N

# Try giving few training examples
NUM_EXAMPLES_FACTOR = 1

class BounceVidCoords(Dataset):

    def __init__(self, feat_dim, batch_size, T=20, subset='train'):
        if subset == 'test':
            print 'test set'
            np.random.seed(432432)
        assert int(np.sqrt(feat_dim)) ** 2 == feat_dim
        super(BounceVidCoords, self).__init__(feat_dim, batch_size)
        self.batch_ind = 0
        self.T = T

    def data_left(self):
        return self.batch_ind < self.feat_dim * NUM_EXAMPLES_FACTOR

    def get_batch(self):
        self.batch = np.zeros((self.feat_dim, self.T, self.batch_size))
        #self.batch_labels = np.zeros((N * 2, self.T, self.batch_size))
        self.batch_labels = np.zeros((self.feat_dim, self.T, self.batch_size))

        # FIXME
        side_len = int(np.sqrt(self.feat_dim))

        for k in xrange(self.batch_size):
            v, c = bounce_vec(side_len, n=N, T=self.T, ret_coords=True)
            v = v.T
            #c = c.transpose((1, 2, 0))
            #c = c.reshape((N*2, -1))
            c2 = np.zeros(v.shape)
            for t in xrange(c2.shape[1]):
                c2[side_len * c[t, 0, 0] + c[t, 0, 1], t] = 1
                c2[side_len * c[t, 1, 0] + c[t, 1, 1], t] = 1

            self.batch[:, 0:self.T, k] = v
            self.batch_labels[:, 0:self.T, k] = c2

        self.batch_ind += 1
        return array(self.batch), array(self.batch_labels)

    def restart(self, shuffle=True):
        self.batch_ind = 0


if __name__ == '__main__':
    # Sanity check
    import matplotlib.pyplot as plt
    vid_coords = BounceVidCoords(400, 10, 20)
    frames, coords = vid_coords.get_batch()
    print frames.shape, coords.shape
    s = int(np.sqrt(vid_coords.feat_dim))
    coords = coords * 2 - 0.4
    for t in xrange(20):
        plt.clf()
        plt.scatter((coords[2, t, 0], coords[0, t, 0]),
                    (coords[3, t, 0], coords[1, t, 0]))
        plt.imshow(frames[:, t, 0].reshape((s, s)))
        plt.show()
