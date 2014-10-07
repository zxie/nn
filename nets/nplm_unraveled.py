import numpy as np
from ops import tanh, mult, rand, zeros, empty
from dset import BrownCorpus
from nag import NesterovOptimizer
from log_utils import get_logger

'''
Implementation of
    "A Neural Probabilistic Language Model",
    Bengio et. al., JMLR 2003
Follows some details given in
    "Decoding with Large-Scale Neural LMs Improves Translation",
    Vaswani et. al., EMNLP 2013
'''

logger = get_logger()

# TODO Handle hyperparameters some other way
embed_size = 30  # size of word embeddings
context_size = 4  # size of word context (so 4 for 5-gram)
hidden_size = 3
# NOTE Each window has a different set of word embeddings to update
batch_size = 512
rand_range = [-0.01, 0.01]

# FIXME
alpha = 1.0

# TODO Move to utils file
class ParamStruct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


class NPLM(object):

    def __init__(self, opt, dset):
        self.opt = opt
        self.params = dict()
        self.dset = dset
        self.vocab_size = len(dset.word_inds)
        logger.debug('Vocab size: %d' % self.vocab_size)

        self.rand_init = lambda shape: rand(shape, rand_range)
        # PARAM Following Vaswani et. al. EMNLP 2013
        self.bias_init = lambda shape: zeros(shape)  #- np.log(self.vocab_size)

        self.alloc_params()
        # Allocate non-parameters
        # Here feat_dim is the context size
        self.C_indexed = empty((self.dset.feat_dim * self.params['C'].shape[0], batch_size))

    def alloc_params(self):
        self.params['C'] = self.rand_init((embed_size, self.vocab_size))
        self.params['H'] = self.rand_init((hidden_size, context_size*embed_size))
        self.params['d'] = self.bias_init((hidden_size, 1))
        self.params['U'] = self.rand_init((self.vocab_size, hidden_size))
        self.params['b'] = self.bias_init((self.vocab_size, 1))
        self.params['W'] = self.rand_init((self.vocab_size, context_size*embed_size))
        logger.info('Allocated parameters')

    def run(self):
        data, labels = self.dset.get_batch()
        cost, grads = self.cost_and_grad(data, labels)
        self.check_grad(data, labels, grads)
        self.update_params(grads)
        return cost

    def cost_and_grad(self, data, labels, back=True):
        p = ParamStruct(**self.params)
        x = self.C_indexed

        for k in range(batch_size):
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
        cost_array = np.empty(batch_size, dtype=np.float64)
        for k in range(batch_size):
            cost_array[k] = -1 * np.log(probs[labels[k], k])
        cost = cost_array.sum() / batch_size

        if not back:
            return cost, None

        # Backprop
        #logger.info('Backprop')

        grads = dict()
        for param in self.params:
            grads[param] = zeros(self.params[param].shape)

        dLdy = probs
        # NOTE This changes probs
        for k in range(batch_size):
            dLdy[labels[k], k] -= 1

        # TODO Move normalization for all grads to the end
        dLdy_sum = dLdy.sum(axis=1) / batch_size
        grads['b'] = dLdy_sum.reshape((-1, 1))

        grads['W'] = mult(dLdy, x.T) / batch_size
        grads['U'] = mult(dLdy, a.T) / batch_size

        dLdx = mult(p.W.T, dLdy)
        dLda = mult(p.U.T, dLdy)

        # NOTE Gradient of nonlinearity is hardcoded here
        dLdo = (1 - a*a) * dLda
        grads['d'] = dLdo.sum(axis=1).reshape((-1, 1)) / batch_size
        dLdx += mult(p.H.T, dLdo)
        grads['H'] = mult(dLdo, x.T) / batch_size
        print data.shape
        for k in range(batch_size):
            for i, l in enumerate(data[:, k]):
                grads['C'][:, l] += dLdx[i*embed_size:(i+1)*embed_size, k] / batch_size

        return cost, grads

    # NOTE Try b eps=1e-3, W eps=10.0, U eps=1e-3, d eps=1e-2, H eps=1.0
    # Turning off large biases (bias_init) also seems to help if get all 0s
    # NOTE For C grad check by checking the columns of words that appear
    # in the current data
    def check_grad(self, data, labels, grads, eps=10):
        #for p in self.params:
        for p in ['C']:
            logger.info('Grad check on %s' % p)
            param = self.params[p]
            grad = grads[p]
            # NOTE Definitely want to use numpy at not 32 bit floats on GPU here
            num_grad = np.empty(param.shape, dtype=np.float64)
            for i in range(param.shape[0]):
                #for j in param.shape[1]:
                ## NOTE Transpose to get words in order
                for j in data.T.ravel():
                    # NOTE Does 2-way numerical gradient
                    param[i, j] += eps
                    cost_p, _ = self.cost_and_grad(data, labels, back=False)
                    param[i, j] -= 2*eps
                    cost_m, _ = self.cost_and_grad(data, labels, back=False)
                    param[i, j] += eps
                    num_grad[i, j] = (cost_p - cost_m) / (2*eps)
                    print 'ng', num_grad[i, j], 'g', grad[i, j], '/', num_grad[i, j] / grad[i, j], '-', num_grad[i, j] - grad[i, j]

    def update_params(self, grads):
        # TODO Replace w/ SGD + mom or NAG
        # TODO Figure out why this allocates more memory
        for p in self.params:
            self.params[p] -= alpha * grads[p]


if __name__ == '__main__':
    # TODO Claim GPU
    # TODO Split into train and test

    # Load dataset
    brown = BrownCorpus(context_size, batch_size)
    nag = NesterovOptimizer()

    # TODO Create optimizer

    # Construct network
    nplm = NPLM(nag, brown)

    # Run training
    epochs = 1
    for k in xrange(0, epochs):
        brown.restart()
        it = 0
        while brown.data_left():
            cost = nplm.run()
            logger.info('epoch %d, iter %d, obj=%f' % (k, it, cost))
            it += 1
