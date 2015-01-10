import random
import numpy as np
import sys
import cPickle as pickle
# to read .xz compressed files
try:
    import lzma
except ImportError:
    from backports import lzma

from dset import Dataset
from dset_paths import CHAR_CORPUS_VOCAB_FILE
from log_utils import get_logger
from preproc_char import CONTEXT
from dset_paths import WEBLM_FILES
'''
For large datasets, can't save the data arrays,
need to allocate on the fly
'''

logger = get_logger()

class SamplingCharStream(Dataset):

    def __init__(self, feat_dim, batch_size, step=1, subset='train'):
        super(SamplingCharStream, self).__init__(feat_dim, batch_size)

        # NOTE Need to specify paths in subclasses
        self.text_path_files = {
                #'train': '/bak/swbd_data/train/files.txt',
                #'test': '/bak/swbd_data/test/files.txt'
                'train': 'dsets/weblm_files.txt',
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

        # how many lines to load at once
        self.line_batch = 1e6
        
        logger.info('Loading lines')
        self.lines = self._load_line_batch(self.line_batch)

    def data_left(self):
        return True

    def _load_line_batch(self,n_lines,min_length=40):
        """
        Opens a random file, samples n_lines of length min_length
        while ignoring any lines with characters not in our vocab
        """
        lines = []
        # prob of keeping a line (lower means more sparsely sampling file
        keep_prob = 1e-1
        vocab_set = set(self.chars.values())
        n_kept = 0
        # pick a random file
        f_name = random.choice(self.files)
        logger.info('Using file %s' % (f_name))
        with lzma.open(f_name,mode="rt") as f:
            # seek a bunch to avoid always reading beginning
            sp = random.randint(0,10)*1e8
            logger.info('Seeking to %d' % (sp))
            f.seek(sp)
            # read to end of current line
            f.readline()
            logger.info('Starting to read lines')
            while n_kept < n_lines:
                l = f.readline()
                if random.random() < keep_prob:
                    # check if chars are only in our vocab
                    l = l.rstrip().encode('ascii', 'ignore')
                    if len(l) >= min_length and len(set(l) - vocab_set) == 0:
                        # print len(set(l) - vocab_set)
                        lines.append(l)
                        n_kept+=1
                        #if n_kept % 1e5 == 0:
                        #print n_kept, ' / ', n_lines
        logger.info('Finished reading file')
        return lines
                        
    # NOTE Assumes that OOV characters have been filtered
    # and text has been lower-cased and stripped
    # gets all data from line and randomly downsamples
    def get_data_from_line(self, line):
        # + 1 since add </s>
        N = len(line) - 1
        char_ind = 0
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
            char_ind += 1

        # randomly permute and keep only a percentage of the data
        ki = range(len(labels))
        ki = ki[1 : int(0.5*len(labels))]
        data = data[:,ki]
        labels = labels[ki]
        return data, labels

    def get_batch(self):
        self.batch = None
        self.batch_labels = None
        while self.batch is None or self.batch.shape[1] < self.batch_size:

            line_text = self.lines[self.line_ind]# .replace('\\', '')
            #print line_text
            line_data, line_labels = self.get_data_from_line(line_text)

            self.line_ind += 1

            if self.line_ind == len(self.lines):
                if self.data_left():
                    # Load next file
                    self.line_ind = 0
                    logger.info('Loading lines')
                    self.lines = self._load_line_batch(self.line_batch)
                else:
                    break

            if self.batch is None:
                self.batch = line_data
                self.batch_labels = line_labels
            else:
                self.batch = np.hstack((self.batch, line_data))
                self.batch_labels = np.concatenate((self.batch_labels, line_labels))

        self.batch = self.batch[:,:self.batch_size]
        self.batch_labels = self.batch_labels[:self.batch_size]
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
    dset = SamplingCharStream(CONTEXT, 512)
    print dset.char_inds
    for k in xrange(5):
        batch, labels = dset.get_batch()
        print batch.shape
        print labels.shape
