import numpy as np
import random
import cPickle as pickle
from char_stream import CharStream
from log_utils import get_logger
from dset_paths import CHAR_CORPUS_VOCAB_FILE

logger = get_logger()
random.seed(19)

# # NOTE Pad at least this much to give the RNN flexibility
MAX_UTT_LENGTH = 300

class UttCharStream(CharStream):

    def __init__(self, batch_size, subset='train'):
        super(CharStream, self).__init__(None, batch_size)

        # NOTE Need to specify paths in subclasses
        self.text_path_files = {
                'train': '/deep/u/zxie/swbd_data/train/files.txt',
                'test': '/deep/u/zxie/swbd_data/test/files.txt'
        }
        self.subset = subset

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
        self.load_file(self.files[self.file_ind])

    def data_left(self):
        return self.file_ind < len(self.files) - 1\
                or self.line_ind < len(self.lines)

    def sort_and_shuffle_lines(self):
        assert len(self.lines) > 0
        logger.info('# lines: %d' % len(self.lines))
        # First shuffle once
        random.shuffle(self.lines)
        # Then sort by length to save computation
        self.lines.sort(lambda x, y: cmp(len(x), len(y)))
        # Then shuffle batches
        batch_inds = list(xrange(len(self.lines) / self.batch_size))
        random.shuffle(batch_inds)
        lines_tmp = self.lines
        self.lines = list()
        for k in batch_inds:
            self.lines.extend(lines_tmp[k*self.batch_size:(k+1)*self.batch_size])
        if len(lines_tmp) % self.batch_size != 0:
            self.lines.extend(lines_tmp[len(batch_inds)*self.batch_size:])

    def get_data_from_line(self, line):
        # NOTE Removing whitespace
        line = line.strip()
        line_chars = [self.char_inds[c] for c in line]
        data = [self.char_inds['<s>']] + line_chars
        labels = data[1:] + [self.char_inds['</s>']]
        return data, labels

    def get_batch(self):
        # NOTE Here the batch contains the batch labels
        self.batch = list()
        self.batch_labels = list()
        while len(self.batch) < self.batch_size:
            # FIXME Next time filter these away beforehand
            line_text = self.lines[self.line_ind].replace('\\', '')
            line_data, line_labels = self.get_data_from_line(line_text)

            self.line_ind += 1

            if self.line_ind == len(self.lines):
                if self.data_left():
                    # Load next file
                    self.line_ind = 0
                    self.file_ind += 1
                    logger.info('Loading %s' % self.files[self.file_ind])
                    self.load_file(self.files[self.file_ind])
                    # Preserve ~same length lines
                    break
                else:
                    break

            self.batch.append(line_data)
            self.batch_labels.append(line_labels)

        return self.batch, self.batch_labels

    def load_file(self, f):
        with open(f, 'r') as fin:
            self.lines = fin.read().splitlines()
            self.lines = [l.strip() for l in self.lines if len(l) < MAX_UTT_LENGTH]
            self.sort_and_shuffle_lines()

    def restart(self, shuffle=True):
        self.file_ind = 0
        self.line_ind = 0
        self.load_file(self.files[self.file_ind])

if __name__ == '__main__':
    # Test things out
    dset = UttCharStream(5)
    print dset.char_inds
    for k in xrange(5):
        batch_data, batch_labels = dset.get_batch()
        print batch_data
        print [len(utt) for utt in batch_data]
