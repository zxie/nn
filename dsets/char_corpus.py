import cPickle as pickle
from dset import Dataset
from dset_paths import CHAR_CORPUS_DATA_FILE, CHAR_CORPUS_VOCAB_FILE
from preproc_char import CONTEXT

class CharCorpus(Dataset):

    def __init__(self, feat_dim, batch_size, subset='train', data_file=CHAR_CORPUS_DATA_FILE):
        super(CharCorpus, self).__init__(feat_dim, batch_size)
        # Load vocab
        with open(CHAR_CORPUS_VOCAB_FILE, 'rb') as fin:
            self.char_inds = pickle.load(fin)
            self.vocab_size = len(self.char_inds)
        self.chars = dict((v, k) for k, v in self.char_inds.iteritems())
        # Load data matrices
        import h5py
        h5f = h5py.File(data_file, 'r')
        self.subset = subset

        # NOTE These are all just word indices, no point
        # in putting them on the GPU
        if subset == 'train':
            self.data = h5f['train'][...]
        elif subset == 'dev':
            self.data = h5f['dev'][...]
        elif subset == 'test':
            self.data = h5f['test'][...]

        assert self.data.shape[0] == (CONTEXT + 1)

        self.labels = self.data[-1, :]

    def get_batch(self):
        self.batch = self.data[:-1, self.data_ind:self.data_ind+self.batch_size]
        self.batch_labels = self.labels[self.data_ind:self.data_ind+self.batch_size]
        self.data_ind += self.batch.shape[1]
        return self.batch, self.batch_labels
