import random
import numpy as np
import cPickle as pickle
from dset import Dataset
from dset_paths import CHAR_CORPUS_VOCAB_FILE
from log_utils import get_logger
from preproc_char import CONTEXT

'''
For large datasets, can't save the data arrays,
need to allocate on the fly
'''

random.seed(1)

logger = get_logger()

class CharStream(Dataset):

    def __init__(self, feat_dim, batch_size, subset='train', context=CONTEXT):
        super(CharStream, self).__init__(feat_dim, batch_size)

        # NOTE Need to specify paths in subclasses
        self.text_path_files = {
                'train': '/bak/swbd_data/train/files.txt',
                'test': '/bak/swbd_data/test/files.txt'
        }
        self.subset = subset
        self.context = context

        # Load vocab
        with open(CHAR_CORPUS_VOCAB_FILE, 'rb') as fin:
            self.char_inds = pickle.load(fin)
            self.vocab_size = len(self.char_inds)
        self.chars = dict((v, k) for k, v in self.char_inds.iteritems())

        # Keep track of where we are
        self.file_ind = 0
        self.line_ind = 0

        assert subset in self.text_path_files
        with open(self.text_path_files[subset], 'r') as f:
            self.files = f.read().strip().split('\n')

        logger.info('Loading %s' % self.files[self.file_ind])
        with open(self.files[self.file_ind], 'r') as fin:
            self.lines = fin.read().splitlines()
            random.shuffle(self.lines)

    def data_left(self):
        return self.file_ind < len(self.files) - 1\
                or self.line_ind < len(self.lines)

    # NOTE Assumes that OOV characters have been filtered
    # and text has been lower-cased and stripped
    def get_batch(self):
        self.batch = np.empty((self.context, self.batch_size), dtype=np.int32)
        self.batch_labels = np.empty((self.context, self.batch_size), dtype=np.int32)

        batch_ind = 0
        context_ind = 0
        while batch_ind < self.batch_size:
            # FIXME Next time filter these away beforehand
            line = self.lines[self.line_ind].replace('\\', '')
            line = [c for c in line if c in self.char_inds]
            line = ['<s>'] + line + ['</s>']

            for k in xrange(len(line) - 1):
                self.batch[context_ind, batch_ind] = self.char_inds[line[k]]
                self.batch_labels[context_ind, batch_ind] = self.char_inds[line[k+1]]
                context_ind += 1
                if context_ind == self.context:
                    context_ind = 0
                    batch_ind += 1
                if batch_ind == self.batch_size:
                    break

            self.line_ind += 1

            if self.line_ind == len(self.lines):
                if self.data_left():
                    # Load next file
                    self.line_ind = 0
                    self.file_ind += 1
                    logger.info('Loading %s' % self.files[self.file_ind])
                    with open(self.files[self.file_ind], 'r') as fin:
                        self.lines = fin.read().splitlines()
                        random.shuffle(self.lines)
                else:
                    break

        return self.batch[:, 0:batch_ind], self.batch_labels[:, 0:batch_ind]

    def restart(self, shuffle=True):
        logger.info('Restarting')
        self.file_ind = 0
        self.line_ind = 0
        with open(self.files[self.file_ind], 'r') as fin:
            self.lines = fin.read().splitlines()
        if shuffle:
            random.shuffle(self.lines)


if __name__ == '__main__':
    # Test things out
    dset = CharStream(CONTEXT, 512)
    print dset.char_inds
    for k in xrange(5):
        batch, labels = dset.get_batch()
        print batch.shape
        print labels.shape
        print batch
        print labels
