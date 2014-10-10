import numpy as np
from ops import tanh, mult, rand, zeros, empty
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

out_file = 'nplm_params.pk'

class NPLMHyperparams(ModelHyperparams):

    def __init__(self, **entries):
        self.defaults = [
            ('embed_size', 30, 'size of word embeddings'),
            ('context_size', 4, 'size of word context (so 4 for 5-gram)'),
            ('hidden_size', 50, 'size of hidden layer'),
            ('batch_size', 512, 'size of dataset batches')
        ]
        super(NPLMHyperparams, self).__init__(entries)


class NPLM(Net):

    def __init__(self, dset, hps, opt_hps, train=True, opt='nag'):
        self.params = dict()
        self.dset = dset
        self.train = train
        self.hps = hps
        self.vocab_size = len(dset.word_inds)
        self.likelihood_size = self.vocab_size
        logger.debug('Vocab size: %d' % self.vocab_size)

        # PARAM
        rand_range = [-0.01, 0.01]
        self.rand_init = lambda shape: rand(shape, rand_range)
        # PARAM Following Vaswani et al. EMNLP 2013
        self.bias_init = lambda shape: zeros(shape) - np.log(self.vocab_size)

        if train:
            self.alloc_params()

            # NOTE Make sure to initialize optimizer after alloc_params
            self.opt = create_optimizer(opt, self, alpha=opt_hps.alpha,
                    mom=opt_hps.mom, mom_low=opt_hps.mom_low,
                    low_mom_iters=opt_hps.low_mom_iters)

    def alloc_params(self):
        hps = self.hps
        self.params['C'] = self.rand_init((hps.embed_size, self.vocab_size))
        self.params['H'] = self.rand_init((hps.hidden_size, hps.context_size*hps.embed_size))
        self.params['d'] = self.bias_init((hps.hidden_size, 1))
        self.params['U'] = self.rand_init((self.vocab_size, hps.hidden_size))
        self.params['b'] = self.bias_init((self.vocab_size, 1))
        self.params['W'] = self.rand_init((self.vocab_size, hps.context_size*hps.embed_size))

        self.param_keys = sorted(self.params.keys())
        logger.info('Allocated parameters')

    def run(self, back=True):
        data, labels = self.dset.get_batch()
        #cost, grads = self.cost_and_grad(data, labels)
        #self.check_grad(data, labels, grads)
        if back:
            self.update_params(data, labels)
        else:
            cost, probs = self.cost_and_grad(data, labels, back=False)
            return cost, probs

    def cost_and_grad(self, data, labels, back=True):
        # May not be full batch size if at end of dataset
        bsize = data.shape[1]
        embed_size = self.hps.embed_size
        p = ParamStruct(**self.params)
        x = empty((self.dset.feat_dim * embed_size, bsize))

        for k in range(bsize):
            # NOTE Transpose to get words in order
            x[:, k] = p.C[:, data[:, k]].T.ravel()

        # Forward prop

        a = tanh(mult(p.H, x) + p.d)
        # TODO Try disabling direct connections, apparently takes more
        # epochs to train but may reach lower perplexity
        y = mult(p.W, x) + mult(p.U, a) + p.b
        # Softmax
        probs = (y - y.max(axis=0)).exp()
        probs = probs / probs.sum(axis=0)

        # NOTE For more precision if necessary convert to nparray early
        cost_array = np.empty(bsize, dtype=np.float64)
        for k in range(bsize):
            cost_array[k] = -1 * np.log(probs[labels[k], k])
        cost = cost_array.sum() / bsize

        if not back:
            return cost, probs

        # Backprop

        grads = dict()
        for param in self.params:
            grads[param] = zeros(self.params[param].shape)

        dLdy = probs
        # NOTE This changes probs
        for k in range(bsize):
            dLdy[labels[k], k] -= 1

        grads['b'] = dLdy.sum(axis=1).reshape((-1, 1))

        grads['W'] = mult(dLdy, x.T)
        grads['U'] = mult(dLdy, a.T)

        dLdx = mult(p.W.T, dLdy)
        dLda = mult(p.U.T, dLdy)

        # NOTE Gradient of nonlinearity is hardcoded here
        dLdo = (1 - a*a) * dLda
        grads['d'] = dLdo.sum(axis=1).reshape((-1, 1))
        dLdx += mult(p.H.T, dLdo)
        grads['H'] = mult(dLdo, x.T)
        for k in range(bsize):
            for i, l in enumerate(data[:, k]):
                grads['C'][:, l] += dLdx[i*embed_size:(i+1)*embed_size, k]

        # Normalize
        for p in grads:
            grads[p] /= bsize

        return cost, grads

    # NOTE Try b eps=1e-3, W eps=10.0, U eps=1e-2, d eps=1e-2, H eps=1.0
    # NOTE For C grad check columns of words that appear in batch
    # in the current data
    def check_grad(self, data, labels, grads, eps=1.0):
        #for p in self.params:
        for p in ['H']:
            logger.info('Grad check on %s' % p)
            param = self.params[p]
            grad = grads[p]
            # NOTE Definitely want to use numpy at not 32 bit floats on GPU here
            num_grad = np.empty(param.shape, dtype=np.float64)
            for i in range(param.shape[0]):
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
