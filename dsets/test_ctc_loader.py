import cPickle as pickle


from ctc_loader import SWBD_TRAIN_ALIGN_FILE

with open('/bak/swbd_data/train/text_cleaned_reordered', 'r') as fin:
    lines = fin.read().splitlines()

with open(SWBD_TRAIN_ALIGN_FILE, 'rb') as fin:
    alignments = pickle.load(fin)

print len(lines)
print len(alignments)

count = 0
for line, alignment in zip(lines, alignments):
    if len(line) != len(alignment):
        print len(line), len(alignment)
        print line
        print alignment
        count += 1

print '%d/%d utts not same length' % (count, len(lines))
