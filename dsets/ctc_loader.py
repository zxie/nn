import numpy as np
import cPickle as pickle
from os.path import join as pjoin
from dset import Dataset
from dset_paths import SWBD_DATA_PATH, DSET_PATH
from log_utils import get_logger
from ops import ones, log, empty, exp, zeros, square, sqrt
import multiprocessing
#import gnumpy as gnp

'''
Very specific class which given alignments and set of ctc
probabilities will load batches of probabilities
'''

# PARAM
CTC_LOGLIKES_DIR = pjoin(DSET_PATH, 'ctc_loglikes_swbd_train')
NUM_LOGLIKE_FILES = 384
SWBD_TRAIN_ALIGN_FILE = pjoin(SWBD_DATA_PATH, 'align.pk')

# PARAM
NUM_CHARS = 35  # NOTE This includes blank
SOURCE_CONTEXT = 15

logger = get_logger()

def uniform_loglikes(n):
    return log(ones((NUM_CHARS, n)) / float(NUM_CHARS))

def blank_loglikes(n):
    a = ones((NUM_CHARS, n)) * 0.1
    a[0, :] = 0.9
    a /= sqrt(square(a).sum(axis=0))
    return log(a)

class CTCLoader(Dataset):

    def __init__(self, feat_dim, batch_size, subset='train'):

        if subset != 'train':
            return

        super(CTCLoader, self).__init__(feat_dim, batch_size)

        with open(SWBD_TRAIN_ALIGN_FILE, 'rb') as fin:
            self.alignments = pickle.load(fin)

        # Keep track of where we are
        self.file_ind = 0
        self.line_ind = 0
        self.char_ind = 0
        self.align_ind = 0

        self.files = list()
        for k in xrange(1, NUM_LOGLIKE_FILES + 1):
            self.files.append(pjoin(CTC_LOGLIKES_DIR, 'loglikelihoods_%d.pk' % k))

        utt_ids_file = pjoin(pjoin(SWBD_DATA_PATH, subset), 'utt_ids')
        logger.info('Loading utt ids from %s' % utt_ids_file)
        with open(utt_ids_file, 'r') as fin:
            self.utt_ids = fin.read().strip().split('\n')

        assert len(self.utt_ids) == len(self.alignments)

        logger.info('Loading %s' % self.files[self.file_ind])
        with open(self.files[self.file_ind], 'r') as fin:
            self.likelihoods = pickle.load(fin)
        if self.file_ind + 1 < len(self.files):
            self.load_file_async(self.file_ind + 1)

    def data_left(self):
        return self.file_ind < len(self.files) - 1\
                or self.align_ind < len(self.utt_ids)

    def get_data_from_line(self, batch_left):
        utt_id = self.utt_ids[self.align_ind]
        align = self.alignments[self.align_ind]
        ll = self.likelihoods[utt_id]

        # + 1 since add </s>, need this to match the batches from CharStream
        N = min(len(align) + 1, batch_left)
        data = empty((self.feat_dim, N))
        for k in xrange(0, N):
            if len(align) > 0:
                a = align[max(self.char_ind-1, 0)]
                llk = ll[:, a:a+SOURCE_CONTEXT]
            else:
                llk = blank_loglikes(1)

            if llk.shape[1] < SOURCE_CONTEXT:
                #llk = gnp.concatenate((llk, uniform_loglikes(SOURCE_CONTEXT - llk.shape[1])), axis=1)
                #llk = np.hstack((llk, uniform_loglikes(SOURCE_CONTEXT - llk.shape[1])))
                llk = np.hstack((llk, blank_loglikes(SOURCE_CONTEXT - llk.shape[1])))

            data[:, k] = llk.ravel()
            self.char_ind += 1

        return data

    def get_batch(self):
        self.batch = None
        while self.batch is None or self.batch.shape[1] < self.batch_size:
            batch_left = self.batch_size if self.batch is None else self.batch_size - self.batch.shape[1]
            line_data = self.get_data_from_line(batch_left)

            # If we've come to end of the line, don't break
            if self.batch is not None and self.batch.shape[1] >= self.batch_size\
                    and self.char_ind < len(self.alignments[self.align_ind]):
                assert self.batch.shape[1] == self.batch_size
                break

            self.char_ind = 0
            self.line_ind += 1
            self.align_ind += 1

            if self.line_ind == len(self.likelihoods):
                if self.data_left():
                    # Load next file
                    self.line_ind = 0
                    self.file_ind += 1
                    self.likelihoods = self.get_data_async()
                    if self.file_ind + 1 < len(self.files):
                        self.load_file_async(self.file_ind + 1)
                else:
                    break

            if self.batch is None:
                self.batch = line_data
            else:
                #self.batch = gnp.concatenate((self.batch, line_data), axis=1)
                self.batch = np.hstack((self.batch, line_data))

        return exp(self.batch)

    def load_file_async(self, file_ind):
        logger.info('Loading %s' % self.files[file_ind])
        self.p_conn, c_conn = multiprocessing.Pipe()
        self.p = multiprocessing.Process(target=self.load_and_pipe_file, args=(file_ind, c_conn))
        self.p.start()

    def load_and_pipe_file(self, file_ind, conn):
        conn.send(self.load_file(file_ind))
        conn.close()

    def load_file(self, file_ind):
        with open(self.files[file_ind], 'rb') as fin:
            likelihoods = pickle.load(fin)
        return likelihoods

    def get_data_async(self):
        likelihoods = self.p_conn.recv()
        self.p.join()
        return likelihoods

    def restart(self, shuffle=True):
        # TODO Multiple files / shuffling
        self.file_ind = 0
        self.line_ind = 0
        self.char_ind = 0
        self.align_ind = 0
        with open(self.files[self.file_ind], 'r') as fin:
            self.likelihoods = pickle.load(fin)
        if self.file_ind + 1 < len(self.files):
            self.load_file_async(self.file_ind + 1)

if __name__ == '__main__':
    loader = CTCLoader(NUM_CHARS*SOURCE_CONTEXT, 512, subset='train')
    for k in range(10):
        data = loader.get_batch()
        print data
        print data.shape
