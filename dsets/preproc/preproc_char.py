import h5py
import cPickle as pickle
import re
import numpy as np
import collections
from os.path import join as pjoin
from dset_paths import BROWN_CORPUS_RAW_PATH, CHAR_CORPUS_DATA_FILE,\
        CHAR_CORPUS_VOCAB_FILE

'''
Based off of preproc_words.py
'''

# FIXME PARAM
CONTEXT = 11


# NOTE
# - For SWBD need ', &, -, /, [space], a-z
# - For WSJ need subset (where [space] -> <SPACE>) but . as well
# - The LM built using web data doesn't cover
#   - {[noise], [vocalized-noise], [laughter]} in the case of SWBD
#   - {<NOISE>} in the case of WSJ
# NOTE For large text corpora do this using C++ or sed as in biglm
pattern = re.compile('[^a-z\-\'\&\/ ]+', re.UNICODE)
def char_filter(text):
    text = pattern.sub('', text)
    return text

def preproc_line(line):
    line = char_filter(line.strip())
    # NOTE Specific to brown corpus, remove tags
    words = ['/'.join(pair.split('/')[0:-1]) if len(pair) > 1 else pair for pair in line.split(' ')]
    line = ' '.join(words)
    line = line.replace(' .', '.')
    return line

# NOTE No need here to worry about adding words
# from the test set that don't appear in training set
def build_vocab(data_files):
    num_chars = 0
    num_sents = 0
    counter = collections.defaultdict(int)

    for f in data_files:
        path = pjoin(BROWN_CORPUS_RAW_PATH, f)
        print 'Processing %s' % path
        with open(path, 'r') as fin:
            data = fin.read().lower().strip().split('\n')
            for line in data:
                line = preproc_line(line)
                if not line:
                    continue
                num_chars += len(line)
                num_sents += 1
                for c in line:
                    counter[c] += 1
    print '%d chars, %d sentences in corpus' % (num_chars, num_sents)
    print '%d chars in "vocab"' % len(counter)
    return sorted(counter.keys()), num_chars, num_sents


def build_data(data_files, train_data, dev_data, test_data, char_inds):
    data_ind = 0
    for f in data_files:
        path = pjoin(BROWN_CORPUS_RAW_PATH, f)
        print 'Processing %s' % path
        with open(path, 'r') as fin:
            data = fin.read().lower().strip().split('\n')
            for line in data:
                line = preproc_line(line)
                if not line:
                    continue
                chars = ['<null>'] * (CONTEXT - 1) + ['<s>'] + list(line) + ['</s>']
                # Begin right after start symbol, stop at end symbol inclusive
                for k in range(CONTEXT, len(chars)):
                    c_inds = [char_inds[c] for c in chars[k-CONTEXT:k+1]]
                    if data_ind < N_train:
                        train_data[:, data_ind] = c_inds
                    elif data_ind < N_train + N_dev:
                        dev_data[:, data_ind - N_train] = c_inds
                    else:
                        test_data[:, data_ind - N_train - N_dev] = c_inds
                    data_ind += 1
    assert data_ind - N_train - N_dev == test_data.shape[1]


if __name__ == '__main__':
    # TODO Swap this out to buid CLM on other datasets
    # Just need to change data_files
    cats_file = pjoin(BROWN_CORPUS_RAW_PATH, 'cats.txt')
    data_files = list()
    with open(cats_file, 'r') as fin:
        lines = fin.read().strip().split('\n')
        for line in lines:
            data_files.append(line.split(' ')[0])

    chars, num_chars, num_sents = build_vocab(data_files)
    print chars

    # Special symbols, no <unk> here
    chars = ['<s>', '</s>', '<null>'] + chars

    char_inds = dict()
    for k in range(len(chars)):
        char_inds[chars[k]] = k

    # PARAM
    N_train = int(num_chars * 0.6)
    N_dev = int(num_chars * 0.2)
    # Add num_sents since we add </s> to every sentence
    N_test = num_chars + num_sents - N_train - N_dev

    train_data = np.empty((CONTEXT + 1, N_train), dtype=np.int8)
    dev_data = np.empty((CONTEXT + 1, N_dev), dtype=np.int8)
    test_data = np.empty((CONTEXT + 1, N_test), dtype=np.int8)

    build_data(data_files, train_data, dev_data, test_data, char_inds)

    # Randomly shuffle and save data

    np.random.seed(19)
    np.random.shuffle(train_data.T)
    np.random.shuffle(dev_data.T)
    np.random.shuffle(test_data.T)

    f = h5py.File(CHAR_CORPUS_DATA_FILE, 'w')
    dset = f.create_dataset('train', train_data.shape, dtype='i8')
    dset[...] = train_data
    dset = f.create_dataset('dev', dev_data.shape, dtype='i8')
    dset[...] = dev_data
    dset = f.create_dataset('test', test_data.shape, dtype='i8')
    dset[...] = test_data
    f.close()

    # Also save vocab

    with open(CHAR_CORPUS_VOCAB_FILE, 'wb') as fout:
        pickle.dump(char_inds, fout)
