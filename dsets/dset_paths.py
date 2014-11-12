import os
from os.path import join as pjoin

if 'DSET_PATH' in os.environ:
    DSET_PATH = os.environ['DSET_PATH']
else:
    DSET_PATH = '/scail/data/group/deeplearning/u/zxie/dsets'

# NLP data

BROWN_CORPUS_URL =\
    'https://ia600503.us.archive.org/21/items/BrownCorpus/brown.zip'
BROWN_CORPUS_RAW_PATH = pjoin(DSET_PATH, 'brown_raw')
BROWN_CORPUS_DATA_FILE = pjoin(DSET_PATH, 'brown_data.h5')
BROWN_CORPUS_VOCAB_FILE = pjoin(DSET_PATH, 'brown_vocab.pk')

#CHAR_CORPUS_DATA_FILE = pjoin(DSET_PATH, 'char_data.h5')
#CHAR_CORPUS_DATA_FILE = pjoin(DSET_PATH, 'swbd_data.h5')
CHAR_CORPUS_VOCAB_FILE = pjoin(DSET_PATH, 'char_vocab.pk')

#SWBD_DATA_PATH = '/scail/group/deeplearning/speech/zxie/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/data'
SWBD_DATA_PATH = '/bak/swbd_data'
SWBD_CORPUS_DATA_FILE = pjoin(DSET_PATH, 'swbd_data.h5')

FSH_DATA_PATH = '/bak/fsh_data'
FSH_CORPUS_DATA_FILE = pjoin(DSET_PATH, 'fsh_data.h5')

CHAR_CORPUS_DATA_FILE = FSH_CORPUS_DATA_FILE
#CHAR_CORPUS_DATA_FILE = SWBD_CORPUS_DATA_FILE

#KENTEXT_FILES = '/bak/dsets_raw/kentext/files.txt'
KENTEXT_FILES = '/bak/dsets_raw/kentext/files_test.txt'
