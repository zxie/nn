import cPickle as pickle
import h5py
import numpy as np
import collections
from os.path import join as pjoin
from dset_paths import BROWN_CORPUS_RAW_PATH, BROWN_CORPUS_DATA_FILE,\
        BROWN_CORPUS_VOCAB_FILE


# FIXME PARAM
N_train = 800000
N_dev = 200000
context = 4

min_count = 4

# NOTE Getting 12918 different words on first 800000 words
# (16384 words on whole corpus) while Bengio et al. report 16383
# Also only get 1161192 total words opposed to 1181041 though...

def word_filter(w):
    return w.isdigit() or w.startswith('$') or w.startswith('**')\
            or w.replace(',', '').isdigit() or w.replace('.', '').isdigit()\
            or w.replace(':', '').isdigit() or w.replace('%', '').isdigit()


def build_vocab(data_files):
    num_words = 0
    num_sents = 0
    counter = collections.defaultdict(int)

    # Only add words in training set to vocab
    words_added = 0
    done = False

    for f in data_files:
        path = pjoin(BROWN_CORPUS_RAW_PATH, f)
        print 'Processing %s' % path
        with open(path, 'r') as fin:
            data = fin.read().strip().split('\n')
            for line in data:
                line = line.strip()
                if not line:
                    continue
                words = [pair.split('/')[0].lower() for pair in line.split(' ')]
                num_words += len(words)
                num_sents += 1
                for word in words:
                    if done or not word:
                        continue
                    counter[word] += 1
                    words_added += 1
                    if words_added >= N_train:
                        done = True
                        break

    num_unk = 0
    keys = counter.keys()
    for word in keys:
        if word_filter(word):
            del counter[word]
        if counter[word] < min_count:
            num_unk += counter[word]
            del counter[word]
    counter['<unk>'] = num_unk

    print '%d words, %d sentences in corpus' % (num_words, num_sents)
    print '%d words in vocab' % len(counter)

    return sorted(counter.keys()), num_words, num_sents


def build_data(data_files, train_data, dev_data, test_data, word_inds):
    context = train_data.shape[0] - 1
    data_ind = 0
    for f in data_files:
        path = pjoin(BROWN_CORPUS_RAW_PATH, f)
        print 'Processing %s' % path
        with open(path, 'r') as fin:
            data = fin.read().strip().split('\n')
            for line in data:
                line = line.strip()
                if not line:
                    continue
                words = [pair.split('/')[0].lower() for pair in line.split(' ')]
                # NOTE Need to be careful here
                words = ['<null>'] * (context - 1) + ['<s>'] + words + ['</s>']
                # Begin right after start symbol, stop at end symbol inclusive
                for k in range(context, len(words)):
                    w_inds = [word_inds[w] for w in words[k-context:k+1]]
                    if data_ind < N_train:
                        train_data[:, data_ind] = w_inds
                    elif data_ind < N_train + N_dev:
                        dev_data[:, data_ind - N_train] = w_inds
                    else:
                        test_data[:, data_ind - N_train - N_dev] = w_inds
                    data_ind += 1
    assert data_ind - N_train - N_dev == test_data.shape[1]


if __name__ == '__main__':
    cats_file = pjoin(BROWN_CORPUS_RAW_PATH, 'cats.txt')
    data_files = list()
    with open(cats_file, 'r') as fin:
        lines = fin.read().strip().split('\n')
        for line in lines:
            data_files.append(line.split(' ')[0])

    vocab, num_words, num_sents = build_vocab(data_files)
    print vocab[0:50]

    # Special symbols, <unk> already handled above
    vocab = ['<s>', '</s>', '<null>'] + vocab
    unk_ind = vocab.index('<unk>')

    word_inds = collections.defaultdict(lambda: unk_ind)

    for k in range(len(vocab)):
        word_inds[vocab[k]] = k
    # Since default dict adds keys when we check it
    word_inds_frozen = dict(word_inds)

    # Add num_sents since we add </s> to every sentence
    N_test = (num_words + num_sents) - N_train - N_dev

    # NOTE Storing 1-hot vectors here extremely in-efficient
    train_data = np.empty((context + 1, N_train), dtype=np.int32)
    dev_data = np.empty((context + 1, N_dev), dtype=np.int32)
    test_data = np.empty((context + 1, N_test), dtype=np.int32)

    build_data(data_files, train_data, dev_data, test_data, word_inds)

    # Randomly shuffle and save data
    # NOTE Shuffling may hurt any future caching you want to do...

    np.random.seed(19)
    np.random.shuffle(train_data.T)
    np.random.shuffle(dev_data.T)
    np.random.shuffle(test_data.T)

    f = h5py.File(BROWN_CORPUS_DATA_FILE, 'w')
    dset = f.create_dataset('train', train_data.shape, dtype='i')
    dset[...] = train_data
    dset = f.create_dataset('dev', dev_data.shape, dtype='i')
    dset[...] = dev_data
    dset = f.create_dataset('test', test_data.shape, dtype='i')
    dset[...] = test_data
    f.close()

    # Also save vocab

    with open(BROWN_CORPUS_VOCAB_FILE, 'wb') as fout:
        pickle.dump(word_inds_frozen, fout)
