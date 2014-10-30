import cPickle as pickle
from os.path import join as pjoin
from dset_paths import CHAR_CORPUS_VOCAB_FILE,\
        FSH_DATA_PATH, FSH_CORPUS_DATA_FILE
from preproc_swbd import preproc_splits

if __name__ == '__main__':
    char_inds = pickle.load(open(CHAR_CORPUS_VOCAB_FILE, 'rb'))

    preproc_splits(pjoin(FSH_DATA_PATH, 'train/text'),
                   pjoin(FSH_DATA_PATH, 'dev/text'),
                   pjoin(FSH_DATA_PATH, 'test/text'),
                   FSH_CORPUS_DATA_FILE)
