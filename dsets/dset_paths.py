from os.path import join as pjoin

DSET_PATH = '/scail/data/group/deeplearning/u/zxie/dsets'

# NLP data

BROWN_CORPUS_URL =\
    'https://ia600503.us.archive.org/21/items/BrownCorpus/brown.zip'
BROWN_CORPUS_RAW_PATH = pjoin(DSET_PATH, 'brown_raw')
BROWN_CORPUS_DATA_FILE = pjoin(DSET_PATH, 'brown_data.h5')
BROWN_CORPUS_VOCAB_FILE = pjoin(DSET_PATH, 'brown_vocab.pk')
