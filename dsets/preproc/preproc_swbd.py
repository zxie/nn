import os
from preproc_char import char_filter
from os.path import join as pjoin
from dset_paths import SWBD_DATA_PATH

'''
Saves SWBD training text in h5 file so we can
test perplexity on it
'''

SPECIALS_LIST = frozenset(['[vocalized-noise]', '[laughter]', '[space]', '[noise]', '(%hesitation)'])

def process_text(text_file):
    num_chars = 0
    transcript = open(text_file, 'r').read().strip()
    lines = transcript.split('\n')
    for k in range(len(lines)):
        lines[k] = ' '.join([w for w in lines[k].lower().split(' ')[1:] if w not in SPECIALS_LIST]).strip()
        lines[k] = char_filter(lines[k])
        num_chars += len(lines[k])
    with open(pjoin(os.path.dirname(text_file), 'text_cleaned'), 'w') as fout:
        fout.write('\n'.join(lines))

def preproc_splits(train_text, dev_text, test_text):
    process_text(train_text)
    print 'Done processing train data'
    process_text(dev_text)
    print 'Done processing dev data'
    process_text(test_text)
    print 'Done processing test data'

if __name__ == '__main__':
    # NOTE Assumes vocab already built, may want to build here later

    preproc_splits(pjoin(SWBD_DATA_PATH, 'train/text_ctc'),
                   pjoin(SWBD_DATA_PATH, 'dev/text_ctc'),
                   pjoin(SWBD_DATA_PATH, 'test/text_ctc'))
