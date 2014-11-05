import argparse
import numpy as np
from optimizer import OptimizerHyperparams
from ops import empty, rand, zeros, get_nl, softmax, mult, get_nl_grad
from models import Net
from log_utils import get_logger
from param_utils import ModelHyperparams
from char_corpus import CharCorpus

# Theano ref: http://stackoverflow.com/questions/24431621/does-theano-do-automatic-unfolding-for-bptt

# Methods
# - BPTT, RTRL, EKF
# TODO
#   - Make sure handling cases at start and end of sequences properly
# - Replace h0 w/ b0
# - Figure out whether ordering should be dim, T, bsize or dim, bsize, T
#   - Need new representation of data w/ time index
# - Start doing some initial training, test perplexity...
# - Will have to clamp / clip gradients
# - Make it bi-directional
# - Deep
# - Arbitrary length / unroll
# - mRNN (hopefully can just subclass and make few changes)

logger = get_logger()

class RNNHyperparams(ModelHyperparams):

    def __init__(self, **entries):
        self.defaults = [
            ('T', 11, 'how much to unroll RNN'),
            ('hidden_size', 10, 'size of hidden layers'),  # FIXME
            ('output_size', 34, 'size of softmax output'),
            ('batch_size', 512, 'size of dataset batches'),
            ('nl', 'relu', 'type of nonlinearity')
        ]
        super(RNNHyperparams, self).__init__(entries)

# PARAM

rand_range = [-0.01, 0.01]
def rand_init(shape):
    return rand(shape, rand_range)

def bias_init(shape):
    return zeros(shape)


class RNN(Net):

    def __init__(self, dset, hps, opt_hps, train=True, opt='nag'):

        super(RNN, self).__init__(dset, hps, train=train)
        self.nl = get_nl(hps.nl)

        self.alloc_params()

    def alloc_params(self):
        # Refer to Ch. 2 pg. 10 of Sutskever's thesis

        hps = self.hps

        # initial hidden state
        # TODO better ways to initialize?
        self.params['h0'] = rand_init((hps.hidden_size, 1))

        # input to hidden, note bias in hidden to hidden
        self.params['Wih'] = rand_init((hps.hidden_size, hps.output_size))

        # hidden to hidden
        self.params['Whh'] = rand_init((hps.hidden_size, hps.hidden_size))
        self.params['bhh'] = bias_init((hps.hidden_size, 1))

        # hidden to output
        self.params['Who'] = rand_init((hps.output_size, hps.hidden_size))
        self.params['bho'] = bias_init((hps.output_size, 1))

        self.count_params()

        # Allocate grads as well

        self.grads = {}
        for k in self.params:
            self.grads[k] = empty(self.params[k].shape)
        logger.info('Allocated gradients')

    def cost_and_grad(self, data, labels, back=True):
        hps = self.hps
        T = hps.T
        # May not be full batch size if at end of dataset
        bsize = data.shape[2]

        h0 = np.repeat(self.params['h0'], bsize, axis=1)
        Wih = self.params['Wih']
        Whh = self.params['Whh']
        bhh = self.params['bhh']
        Who = self.params['Who']
        bho = self.params['bho']

        # Intermediate parameters and their grads
        # TODO May want to allocate once elsewhere

        us = empty((hps.hidden_size, T, bsize))
        dus = zeros((hps.hidden_size, T, bsize))
        hs = empty((hps.hidden_size, T, bsize))
        dhs = zeros((hps.hidden_size, T, bsize))
        costs = empty((T, bsize))
        probs = empty((hps.output_size, T, bsize))
        dprobs = empty((hps.output_size, T, bsize))

        # Forward prop

        for t in xrange(T):
            if t == 0:
                hprev = h0
            else:
                hprev = hs[:, t-1, :]
            us[:, t, :] = mult(Wih, data[:, t, :]) + mult(Whh, hprev) + bhh
            hs[:, t, :] = self.nl(us[:, t, :])
            probs[:, t, :] = softmax(mult(Who, hs[:, t, :]) + bho)
            for k in xrange(bsize):
                costs[t, k] = -1 * np.log(probs[labels[t, k], t, k])

        if labels is None:
            return None, probs

        # NOTE Summing costs over time
        cost = costs.sum() / bsize
        if not back:
            return cost, probs

        # Backprop

        dprobs = probs
        for k in self.grads:
            self.grads[k][:] = 0
        for t in xrange(T):
            for k in xrange(bsize):
                dprobs[labels[t, k], t, k] -= 1
        for t in reversed(xrange(T)):
            self.grads['bho'] += np.mean(dprobs[:, t, :], axis=-1).reshape((-1, 1))
            self.grads['Who'] += mult(dprobs[:, t, :], hs[:, t, :].T) / bsize
            dhs[:, t, :] += mult(Who.T, dprobs[:, t, :])
            dus[:, t, :] += get_nl_grad(self.hps.nl, us[:, t, :]) * dhs[:, t, :]
            self.grads['Wih'] += mult(dus[:, t, :], data[:, t, ].T) / bsize
            self.grads['bhh'] += np.mean(dus[:, t, :], axis=-1).reshape((-1, 1))
            if t == 0:
                hprev = h0
                hgrad = self.grads['h0']
            else:
                hprev = hs[:, t-1, :]
                hgrad = dhs[:, t-1, :]
            self.grads['Whh'] += mult(dus[:, t, :], hprev.T) / bsize
            hgrad[:] = np.mean(mult(Whh.T, dus[:, t, :]), axis=-1).reshape((-1, 1))

        return cost, self.grads


def one_hot(data, n):
    data_1h = zeros((n, data.shape[0], data.shape[1]))
    for t in xrange(data.shape[0]):
        for b in xrange(data.shape[1]):
            data_1h[data[t, b], t, b] = 1
    logger.info('Data shape: %s' % str(data_1h.shape))
    return data_1h


def label_seq(data, final_labels):
    '''
    Use data and labels at final time step to build labels
    at every time step for RNN training
    '''
    labels = empty((data.shape[0], data.shape[1]))
    for t in xrange(1, data.shape[0]):
        labels[t-1, :] = data[t, :]
    labels[-1, :] = final_labels
    logger.info('Labels shape: %s' % str(labels.shape))
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    model_hps = RNNHyperparams()
    opt_hps = OptimizerHyperparams()
    model_hps.add_to_argparser(parser)
    opt_hps.add_to_argparser(parser)

    args = parser.parse_args()

    model_hps.set_from_args(args)
    opt_hps.set_from_args(args)

    dset = CharCorpus(args.T, args.batch_size)

    # Construct network
    model = RNN(dset, model_hps, opt_hps, opt='nag')

    data, labels = dset.get_batch()
    labels = label_seq(data, labels)
    data = one_hot(data, model_hps.output_size)
    cost, grads = model.cost_and_grad(data, labels)
    model.check_grad(data, labels, grads, params_to_check=['Whh'], eps=0.01)
