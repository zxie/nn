from os.path import join as pjoin
from dset_paths import FSH_DATA_PATH
from preproc_swbd import preproc_splits

if __name__ == '__main__':
    preproc_splits(pjoin(FSH_DATA_PATH, 'train/text'),
                   pjoin(FSH_DATA_PATH, 'dev/text'),
                   pjoin(FSH_DATA_PATH, 'test/text'))
