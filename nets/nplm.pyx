import cython
import numpy as np
cimport numpy as np
from ops import mult, rand, zeros, exp, empty, get_nl,\
        get_nl_grad, softmax
from param_utils import ParamStruct, ModelHyperparams
from log_utils import get_logger
from opt_utils import create_optimizer
from models import Net

'''
Implementation of
    "A Neural Probabilistic Language Model",
    Bengio et al., JMLR 2003
Follows some details given in
    "Decoding with Large-Scale Neural LMs Improves Translation",
    Vaswani et al., EMNLP 2013
'''

logger = get_logger()

class NPLMHyperparams(ModelHyperparams):

    def __init__(self, **entries):
        self.defaults = [
            ('embed_size', 30, 'size of word embeddings'),
            ('T', 4, 'size of word context (so 4 for 5-gram)'),
            ('hidden_size', 50, 'size of hidden layer'),
            ('batch_size', 512, 'size of dataset batches'),
            # Not really a hyperparameter...
            ('nl', 'relu', 'type of nonlinearity')
        ]
        super(NPLMHyperparams, self).__init__(entries)


# Moved outside class so Cython + multithreading works

# PARAM

rand_range = [-0.1, 0.1]
def rand_init(shape):
    return rand(shape, rand_range)

def bias_init(shape):
    return zeros(shape)


class NPLM(Net):

    def __init__(self, dset, hps, opt_hps, train=True, opt='nag'):
        super(NPLM, self).__init__(dset, hps, train=train)
        self.nl = get_nl(hps.nl)
        self.vocab_size = dset.vocab_size
        self.likelihood_size = self.vocab_size
        logger.debug('Vocab size: %d' % self.vocab_size)

        self.alloc_params()

        if train:
            # NOTE Make sure to initialize optimizer after alloc_params
            self.opt = create_optimizer(opt, self, alpha=opt_hps.alpha,
                    mom=opt_hps.mom, mom_low=opt_hps.mom_low,
                    low_mom_iters=opt_hps.low_mom_iters)

    def alloc_params(self):
        hps = self.hps
        self.params['C'] = rand_init((hps.embed_size, self.vocab_size))
        self.params['H'] = rand_init((hps.hidden_size, hps.T*hps.embed_size))
        self.params['d'] = bias_init((hps.hidden_size, 1))
        self.params['U'] = rand_init((self.vocab_size, hps.hidden_size))
        self.params['b'] = bias_init((self.vocab_size, 1))
        self.params['W'] = rand_init((self.vocab_size, hps.T*hps.embed_size))

        self.count_params()

    def run(self, back=True):
        super(NPLM, self).run(back=back)

        data, labels = self.dset.get_batch()
        #cost, grads = self.cost_and_grad(data, labels)
        #self.check_grad(data, labels, grads)
        if back:
            self.update_params(data, labels)
        else:
            cost, probs = self.cost_and_grad(data, labels, back=False)
            return cost, probs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def cost_and_grad(self, data, labels, back=True):
        # May not be full batch size if at end of dataset
        bsize = data.shape[1]
        embed_size = self.hps.embed_size
        p = ParamStruct(**self.params)
        x = empty((self.dset.feat_dim * embed_size, bsize))

        cdef int k, j
        cdef int context_size = self.hps.T

        for k in xrange(bsize):
            # NOTE Transpose to get words in order
            x[:, k] = p.C[:, data[:, k]].T.ravel()

        # Forward prop

        a = self.nl(mult(p.H, x) + p.d)
        # TODO Try disabling direct connections, apparently takes more
        # epochs to train but may reach lower perplexity
        y = mult(p.W, x) + mult(p.U, a) + p.b
        # Softmax
        probs = softmax(y)

        if labels is None:
            return None, probs

        # NOTE For more precision if necessary convert to nparray early
        cost_array = np.empty(bsize, dtype=np.float64)
        for k in xrange(bsize):
            cost_array[k] = -1 * np.log(probs[labels[k], k])
        cost = cost_array.sum() / bsize

        if not back:
            return cost, probs

        # Backprop

        # FIXME Allocate in alloc_params()
        grads = dict()
        for param in self.params:
            grads[param] = zeros(self.params[param].shape)

        dLdy = probs
        # NOTE This changes probs
        for k in xrange(bsize):
            dLdy[labels[k], k] -= 1

        grads['b'] = dLdy.sum(axis=1).reshape((-1, 1))

        grads['W'] = mult(dLdy, x.T)
        grads['U'] = mult(dLdy, a.T)

        dLdx = mult(p.W.T, dLdy)
        dLda = mult(p.U.T, dLdy)

        dLdo = get_nl_grad(self.hps.nl, a) * dLda
        grads['d'] = dLdo.sum(axis=1).reshape((-1, 1))
        dLdx += mult(p.H.T, dLdo)
        grads['H'] = mult(dLdo, x.T)
        for k in xrange(bsize):
            #for i, l in enumerate(data[:, k]):
                #grads['C'][:, l] += dLdx[i*embed_size:(i+1)*embed_size, k]
            for j in xrange(context_size):
                grads['C'][:, data[j, k]] += dLdx[j*embed_size:(j+1)*embed_size, k]

        # Normalize
        for p in grads:
            grads[p] /= bsize

        return cost, grads

    # NOTE Try b eps=1e-3, W eps=10.0, U eps=1e-2, d eps=1e-2, H eps=1.0
    # NOTE For C grad check columns of words that appear in batch
    # in the current data
    def check_grad(self, data, labels, grads, eps=0.01):
        #for p in self.params:
        cdef int i, j
        for p in ['W']:
            logger.info('Grad check on %s' % p)
            param = self.params[p]
            grad = grads[p]
            # NOTE Want to use numpy at not 32 bit floats on GPU here
            num_grad = np.empty(param.shape, dtype=np.float64)
            for i in xrange(param.shape[0]):
                ## NOTE Transpose to get words in order
                #for j in data.T.ravel():
                for j in range(param.shape[1]):
                    # NOTE Does 2-way numerical gradient
                    param[i, j] += eps
                    cost_p, _ = self.cost_and_grad(data, labels, back=False)
                    param[i, j] -= 2*eps
                    cost_m, _ = self.cost_and_grad(data, labels, back=False)
                    param[i, j] += eps
                    num_grad[i, j] = (cost_p - cost_m) / (2*eps)
                    print 'ng', num_grad[i, j], 'g', grad[i, j], '/', num_grad[i, j] / grad[i, j], '-', num_grad[i, j] - grad[i, j]
