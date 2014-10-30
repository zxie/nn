import numpy as np
import h5py
import cPickle as pickle
from preproc_char import char_filter, context
from os.path import join as pjoin
from dset_paths import CHAR_CORPUS_VOCAB_FILE,\
    SWBD_DATA_PATH, SWBD_CORPUS_DATA_FILE

'''
Saves SWBD training text in h5 file so we can
test perplexity on it
'''

SPECIALS_LIST = frozenset(['[vocalized-noise]', '[laughter]', '[space]', '[noise]', '(%hesitation)'])

def process_text(text_file, char_inds):
    num_chars = 0
    transcript = open(text_file, 'r').read().strip()
    lines = transcript.split('\n')
    for k in range(len(lines)):
        lines[k] = ' '.join([w for w in lines[k].lower().split(' ')[1:] if w not in SPECIALS_LIST]).strip()
        lines[k] = char_filter(lines[k])
        num_chars += len(lines[k])

    # Add num sents since we add </s> to every sentence
    N_data = num_chars + len(lines)
    print 'N_data: %d' % N_data
    data = np.empty((context + 1, N_data), dtype=np.int8)

    data_ind = 0
    for line in lines:
        chars = ['<null>'] * (context - 1) + ['<s>'] + list(line) + ['</s>']

        for k in range(context, len(chars)):
            c_inds = [char_inds[c] for c in chars[k-context:k+1]]
            data[:, data_ind] = c_inds
            data_ind += 1
    assert data_ind == N_data

    return data

def preproc_splits(train_text, dev_text, test_text, out_file):
    char_inds = pickle.load(open(CHAR_CORPUS_VOCAB_FILE, 'rb'))

    train_data = process_text(train_text, char_inds)
    print 'Done processing train data'
    dev_data = process_text(dev_text, char_inds)
    print 'Done processing dev data'
    test_data = process_text(test_text, char_inds)
    print 'Done processing test data'

    print 'Shuffling...'
    np.random.seed(19)
    np.random.shuffle(train_data.T)
    np.random.shuffle(dev_data.T)
    np.random.shuffle(test_data.T)

    f = h5py.File(out_file, 'w')
    dset = f.create_dataset('train', train_data.shape, dtype='i8')
    dset[...] = train_data
    dset = f.create_dataset('dev', dev_data.shape, dtype='i8')
    dset[...] = dev_data
    dset = f.create_dataset('test', test_data.shape, dtype='i8')
    dset[...] = test_data
    f.close()


if __name__ == '__main__':
    # NOTE Assumes vocab already built, may want to build here later

    preproc_splits(pjoin(SWBD_DATA_PATH, 'train/text'),
                   pjoin(SWBD_DATA_PATH, 'dev/text'),
                   pjoin(SWBD_DATA_PATH, 'test/text'),
                   SWBD_CORPUS_DATA_FILE)
