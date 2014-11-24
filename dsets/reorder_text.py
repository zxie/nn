from os.path import join as pjoin
import cPickle as pickle
from dset_paths import SWBD_DATA_PATH
from ctc_loader import CTC_LOGLIKES_DIR, NUM_LOGLIKE_FILES

'''
Reorder swbd training text so that utterances appear
in the same order as they appear in log-likelihoods
'''


'''
with open('/bak/swbd_data/train/text_ctc', 'r') as fin:
    utt_ids = [line.split(' ')[0] for line in fin.read().strip().split('\n')]
with open('/bak/swbd_data/train/text_cleaned', 'r') as fin:
    texts = fin.read().strip().split('\n')
assert len(utt_ids) == len(texts)
utt_texts = dict(zip(utt_ids, texts))

files = list()
for k in xrange(1, NUM_LOGLIKE_FILES + 1):
    files.append(pjoin(CTC_LOGLIKES_DIR, 'loglikelihoods_%d.pk' % k))

utt_ids_file = pjoin(SWBD_DATA_PATH, 'train/utt_ids')
text_reordered_file = pjoin(SWBD_DATA_PATH, 'train/text_cleaned_reordered')
ufout = open(utt_ids_file, 'w')
tfout = open(text_reordered_file, 'w')
for f in files:
    print f
    with open(f, 'rb') as fin:
        ll_keys = sorted(pickle.load(fin).keys())
    for key in ll_keys:
        ufout.write(key + '\n')
        tfout.write(utt_texts[key] + '\n')
ufout.close()
tfout.close()
'''

'''
Better idea: just read from refs in pickle file
Need to write the utt ids above first though...
'''

with open(pjoin(SWBD_DATA_PATH, 'train.pk'), 'rb') as fin:
    _ = pickle.load(fin)
    refs = pickle.load(fin)

utt_ids_file = pjoin(SWBD_DATA_PATH, 'train/utt_ids')
text_reordered_file = pjoin(SWBD_DATA_PATH, 'train/text_cleaned_reordered')
tfout = open(text_reordered_file, 'w')
for ref in refs:
    s = ''.join([c if c != '[space]' else ' ' for c in ref])
    tfout.write(s + '\n')
tfout.close()
