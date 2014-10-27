import math
import h5py
import argparse
#from brown_corpus import BrownCorpus
from char_corpus import CharCorpus

'''
Compute perplexity given a set of likelihoods and labels
'''

def compute_pp(likelihoods, labels):
    # Working in log space
    pp = 0.0

    N = likelihoods.shape[1]
    assert N == labels.size
    for k in xrange(N):
        pp = pp - math.log(likelihoods[labels[k], k])
    pp = pp / N

    pp = math.exp(pp)

    return pp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('likelihoods_file', help='File containing likelihoods')
    args = parser.parse_args()

    # NOTE These parameters don't affect calculation
    context_size = 4
    batch_size = 512
    #dataset = BrownCorpus(context_size, batch_size, subset='dev')
    dataset = CharCorpus(context_size, batch_size, subset='test')
    labels = dataset.labels

    h5f = h5py.File(args.likelihoods_file)
    likelihoods = h5f['likelihoods'][...]

    pp = compute_pp(likelihoods, labels)

    print 'Perplexity: %f' % pp
    print 'Bits/unit: %f' % math.log(pp, 2)
