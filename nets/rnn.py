import numpy as np
from optimizer import OptimizerHyperparams
from ops import empty, zeros, get_nl, softmax, mult, tile,\
        get_nl_grad, as_np, array, log, vp_init
from models import Net
from log_utils import get_logger
from param_utils import ModelHyperparams
from char_corpus import CharCorpus, CONTEXT
from opt_utils import create_optimizer
from dset_utils import one_hot

# NOTE Theano ref: http://stackoverflow.com/questions/24431621/does-theano-do-automatic-unfolding-for-bptt
# NOTE Currently using BPTT, there'as also RTRL, EKF
# NOTE Switching time and batch index (d, b, t) seems to be slower than current (d, t, b)

# H5 files should have attributes that allow you to automatically determine stuff...

# TODO
# - Gradient clipping
# - Bi-directional
# - Deep
# - Max length / unroll instead of fixed?
# - mRNN (hopefully can just subclass and make few changes)
# - Need to figure out best nonlinearities too

# Maybe
# - Replace h0 w/ b0

logger = get_logger()

class RNNHyperparams(ModelHyperparams):

    def __init__(self, **entries):
        self.defaults = [
            ('T', CONTEXT, 'how much to unroll RNN'),
            ('hidden_size', 800, 'size of hidden layers'),
            ('output_size', 34, 'size of softmax output'),
            ('batch_size', 512, 'size of dataset batches'),
            ('nl', 'relu', 'type of nonlinearity')
        ]
        super(RNNHyperparams, self).__init__(entries)

# PARAM

class RNN(Net):

    def __init__(self, dset, hps, opt_hps, train=True, opt='nag'):

        super(RNN, self).__init__(dset, hps, train=train)
        self.nl = get_nl(hps.nl)

        self.alloc_params()

        if train:
            self.opt = create_optimizer(opt, self, alpha=opt_hps.alpha,
                    mom=opt_hps.mom, mom_low=opt_hps.mom_low,
                    low_mom_iters=opt_hps.low_mom_iters)

    @staticmethod
    def init_hyperparams():
        return RNNHyperparams()

    def alloc_params(self):
        # Refer to Ch. 2 pg. 10 of Sutskever's thesis

        hps = self.hps

        # initial hidden state
        # TODO better ways to initialize?
        self.params['h0'] = zeros((hps.hidden_size, 1))

        # input to hidden, note bias in hidden to hidden
        self.params['Wih'] = vp_init((hps.hidden_size, hps.output_size))

        # hidden to hidden
        self.params['Whh'] = vp_init((hps.hidden_size, hps.hidden_size))
        self.params['bhh'] = zeros((hps.hidden_size, 1))

        # hidden to output
        self.params['Who'] = vp_init((hps.output_size, hps.hidden_size))
        self.params['bho'] = zeros((hps.output_size, 1))

        self.count_params()

        # Allocate grads as well

        self.grads = {}
        for k in self.params:
            self.grads[k] = empty(self.params[k].shape)
        logger.info('Allocated gradients')

    def run(self, back=True):
        super(RNN, self).run(back=back)

        data, labels = self.dset.get_batch()
        labels = label_seq(data, labels)
        data = one_hot(data, self.hps.output_size)
        #cost, grads = self.cost_and_grad(data, labels)
        #self.check_grad(data, labels, grads, params_to_check=['bhh'])
        if back:
            self.update_params(data, labels)
        else:
            cost, probs = self.cost_and_grad(data, labels, back=False)
            return cost, probs

    #@profile
    def cost_and_grad(self, data, labels, back=True):
        hps = self.hps
        T = hps.T
        # May not be full batch size if at end of dataset
        bsize = data.shape[2]

        h0 = tile(self.params['h0'], bsize)
        Wih = self.params['Wih']
        Whh = self.params['Whh']
        bhh = self.params['bhh']
        Who = self.params['Who']
        bho = self.params['bho']

        # Intermediate parameters and their grads
        # TODO May want to allocate once elsewhere, but
        # from profiling doesn't take up much time

        us = empty((hps.hidden_size, T, bsize))
        dus = zeros((hps.hidden_size, T, bsize))
        hs = empty((hps.hidden_size, T, bsize))
        dhs = zeros((hps.hidden_size, T, bsize))
        probs = empty((hps.output_size, T, bsize))
        dprobs = empty((hps.output_size, T, bsize))
        costs = np.empty((T, bsize))

        # Forward prop

        for t in xrange(T):
            if t == 0:
                hprev = h0
            else:
                hprev = hs[:, t-1, :]
            us[:, t, :] = mult(Wih, data[:, t, :]) + mult(Whh, hprev) + bhh
            hs[:, t, :] = self.nl(us[:, t, :])
            probs[:, t, :] = softmax(mult(Who, hs[:, t, :]) + bho)

        if labels is None:
            return None, probs

        probs_neg_log = as_np(-1 * log(probs))
        for t in xrange(T):
            for k in xrange(bsize):
                costs[t, k] = probs_neg_log[labels[t, k], t, k]

        # NOTE Summing costs over time
        cost = costs.sum() / bsize
        if not back:
            return cost, probs

        # Backprop

        dprobs = as_np(probs)
        for k in self.grads:
            self.grads[k][:] = 0
        for t in xrange(T):
            for k in xrange(bsize):
                dprobs[labels[t, k], t, k] -= 1
        dprobs = array(dprobs)
        for t in reversed(xrange(T)):
            self.grads['bho'] += dprobs[:, t, :].sum(axis=-1).reshape((-1, 1)) / bsize
            self.grads['Who'] += mult(dprobs[:, t, :], hs[:, t, :].T) / bsize
            dhs[:, t, :] += mult(Who.T, dprobs[:, t, :])
            dus[:, t, :] += get_nl_grad(self.hps.nl, us[:, t, :]) * dhs[:, t, :]
            self.grads['Wih'] += mult(dus[:, t, :], data[:, t, ].T) / bsize
            self.grads['bhh'] += dus[:, t, :].sum(axis=-1).reshape((-1, 1)) / bsize
            if t == 0:
                hprev = h0
                hgrad = self.grads['h0']
            else:
                hprev = hs[:, t-1, :]
                hgrad = dhs[:, t-1, :]
            self.grads['Whh'] += mult(dus[:, t, :], hprev.T) / bsize
            hgrad[:] = mult(Whh.T, dus[:, t, :]).sum(axis=-1).reshape(-1, 1) / bsize

        return cost, self.grads


def label_seq(data, final_labels):
    '''
    Use data and labels at final time step to build labels
    at every time step for RNN training
    '''
    # NOTE Not putting labels on GPU
    labels = np.empty((data.shape[0], data.shape[1]))
    for t in xrange(1, data.shape[0]):
        labels[t-1, :] = data[t, :]
    labels[-1, :] = final_labels
    #logger.info('Labels shape: %s' % str(labels.shape))
    return labels

if __name__ == '__main__':
    import argparse
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
    model.run()
