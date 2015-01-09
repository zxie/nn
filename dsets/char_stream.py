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

logger = get_logger()

class CharStream(Dataset):

    def __init__(self, feat_dim, batch_size, step=1, subset='train'):
        super(CharStream, self).__init__(feat_dim, batch_size)

        # NOTE Need to specify paths in subclasses
        self.text_path_files = {
                #'train': '/bak/swbd_data/train/files.txt',
                #'test': '/bak/swbd_data/test/files.txt'
                'train': '/deep/u/zxie/swbd_data/train/files.txt',
                'test': '/deep/u/zxie/swbd_data/test/files.txt'
        }
        self.subset = subset
        self.step = step

        # Load vocab
        with open(CHAR_CORPUS_VOCAB_FILE, 'rb') as fin:
            self.char_inds = pickle.load(fin)
            self.vocab_size = len(self.char_inds)
        self.chars = dict((v, k) for k, v in self.char_inds.iteritems())

        # Keep track of where we are
        self.file_ind = 0
        self.line_ind = 0
        self.char_ind = 0

        assert subset in self.text_path_files
        with open(self.text_path_files[subset], 'r') as f:
            self.files = f.read().strip().split('\n')

        logger.info('Loading %s' % self.files[self.file_ind])
        with open(self.files[self.file_ind], 'r') as fin:
            self.lines = fin.read().splitlines()

    def data_left(self):
        return self.file_ind < len(self.files) - 1\
                or self.line_ind < len(self.lines)

    # NOTE Assumes that OOV characters have been filtered
    # and text has been lower-cased and stripped
    def get_data_from_line(self, line, batch_left):
        # + 1 since add </s>
        N = min(len(line) - self.char_ind + 1, batch_left * self.step)
        line = ['<null>'] * (CONTEXT - 1) + ['<s>'] + list(line) + ['</s>']
        data = np.empty((CONTEXT, N/self.step), dtype=np.int32)
        labels = np.empty(N/self.step, dtype=np.int32)
        for k in xrange(0, N-(N % self.step), self.step):
            #print k, self.step
            j = k / self.step
            if k > 0 and self.step == 1:
                data[:-1, k] = data[1:, k-1]
                data[-1, k] = labels[k-1]
            else:
                data[:, j] = [self.char_inds[c] for c in line[k:k+CONTEXT]]
            labels[j] = self.char_inds[line[k+CONTEXT]]
            self.char_ind += 1

        return data, labels

    def get_batch(self):
        self.batch = None
        self.batch_labels = None
        while self.batch is None or self.batch.shape[1] < self.batch_size:
            batch_left = self.batch_size if self.batch is None else self.batch_size - self.batch.shape[1]
            # FIXME Next time filter these away beforehand
            line_text = self.lines[self.line_ind].replace('\\', '')
            #print line_text
            line_data, line_labels = self.get_data_from_line(line_text, batch_left)

            # If we've come to end of the line, don't break
            if self.batch is not None and self.batch.shape[1] >= self.batch_size\
                    and self.char_ind < len(line_text):
                assert batch.shape[1] == self.batch_size
                break

            self.char_ind = 0
            self.line_ind += 1

            if self.line_ind == len(self.lines):
                if self.data_left():
                    # Load next file
                    self.line_ind = 0
                    self.file_ind += 1
                    logger.info('Loading %s' % self.files[self.file_ind])
                    with open(self.files[self.file_ind], 'r') as fin:
                        self.lines = fin.read().splitlines()
                else:
                    break

            if self.batch is None:
                self.batch = line_data
                self.batch_labels = line_labels
            else:
                self.batch = np.hstack((self.batch, line_data))
                self.batch_labels = np.concatenate((self.batch_labels, line_labels))

        return self.batch, self.batch_labels

    def restart(self, shuffle=False):
        # Don't shuffle over utts, probably way too slow
        if shuffle:
            random.shuffle(self.files)
        self.file_ind = 0
        self.line_ind = 0
        self.char_ind = 0
        with open(self.files[self.file_ind], 'r') as fin:
            self.lines = fin.read().splitlines()


if __name__ == '__main__':
    # Test things out
    dset = CharStream(CONTEXT, 512)
    print dset.char_inds
    for k in xrange(5):
        batch, labels = dset.get_batch()
        print batch.shape
        print labels.shape
