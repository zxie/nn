import math
import h5py
import argparse

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

    h5f = h5py.File(args.likelihoods_file)
    likelihoods = h5f['likelihoods'][...]
    labels = h5f['labels'][...]

    pp = compute_pp(likelihoods, labels)

    print 'Perplexity: %f' % pp
    print 'Bits/unit: %f' % math.log(pp, 2)
