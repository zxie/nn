import numpy as np
from optimizer import OptimizerHyperparams
from ops import empty, zeros, get_nl, softmax, mult, tile,\
        get_nl_grad, as_np, array, log, vp_init, rand,\
        USE_GPU, gnp, mean
from models import Net
from log_utils import get_logger
from param_utils import ModelHyperparams
from utt_char_stream import UttCharStream, MAX_UTT_LENGTH
from opt_utils import create_optimizer
from dset_utils import one_hot_lists

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
            ('hidden_size', 2200, 'size of hidden layers'),
            ('output_size', 34, 'size of softmax output'),
            ('batch_size', 128, 'size of dataset batches'),
            ('max_grad', 1000.0, 'threshold to perform gradient clipping'),
            ('nl', 'relu', 'type of nonlinearity')
        ]
        super(RNNHyperparams, self).__init__(entries)

# PARAM

INIT_EPS = 0.01

class RNN(Net):

    def __init__(self, dset, hps, opt_hps, train=True, opt='nag'):

        super(RNN, self).__init__(dset, hps, train=train)
        self.nl = get_nl(hps.nl)

        self.alloc_params()
        self.alloc_grads()

        if train:
            self.opt = create_optimizer(opt, self, alpha=opt_hps.alpha,
                    mom=opt_hps.mom, mom_low=opt_hps.mom_low,
                    low_mom_iters=opt_hps.low_mom_iters, max_grad=hps.max_grad)

    @staticmethod
    def init_hyperparams():
        return RNNHyperparams()

    def alloc_params(self):
        # Refer to Ch. 2 pg. 10 of Sutskever's thesis

        hps = self.hps

        # initial hidden state
        # TODO better ways to initialize?
        self.params['h0'] = rand((hps.hidden_size, 1), rg=[-INIT_EPS, INIT_EPS])
        #self.params['h0'] = zeros((hps.hidden_size, 1))

        # input to hidden, note bias in hidden to hidden
        self.params['Wih'] = vp_init((hps.hidden_size, hps.output_size))

        # hidden to hidden
        # NOTE Initialization important for grad check, don't use vp_init
        #self.params['Whh'] = vp_init((hps.hidden_size, hps.hidden_size))
        self.params['Whh'] = rand((hps.hidden_size, hps.hidden_size), rg=[-INIT_EPS, INIT_EPS])
        self.params['bhh'] = zeros((hps.hidden_size, 1))

        # hidden to output
        self.params['Who'] = vp_init((hps.output_size, hps.hidden_size))
        self.params['bho'] = zeros((hps.output_size, 1))

        # Keep around last hidden state in case want to resume RNN from there
        self.last_h = None

        self.count_params()

    def run(self, back=True, check_grad=False):
        if USE_GPU:
            gnp.free_reuse_cache()
        super(RNN, self).run(back=back)

        data, labels = self.dset.get_batch()
        data = one_hot_lists(data, self.hps.output_size)

        if check_grad:
            cost, grads = self.cost_and_grad(data, labels)
            self.check_grad(data, labels, grads, params_to_check=['h0'], eps=0.01)

        if back:
            self.update_params(data, labels)
        else:
            cost, probs = self.cost_and_grad(data, labels, back=False)
            return cost, probs

    def cost_and_grad(self, data, labels, back=True, prev_h0=None):
        hps = self.hps
        T = data.shape[1]
        #logger.info('T: %d' % T)
        # May not be full batch size if at end of dataset
        bsize = data.shape[2]

        #us = self.us[:, 0:T, 0:bsize]
        #dus = self.dus[:, 0:T, 0:bsize]
        #hs = self.hs[:, 0:T, 0:bsize]
        #dhs = self.dhs[:, 0:T, 0:bsize]
        #probs = self.probs[:, 0:T, 0:bsize]
        #dprobs = self.dprobs[:, 0:T, 0:bsize]
        #costs = self.costs[0:T, 0:bsize]
        us = zeros((hps.hidden_size, T, bsize))
        dus = zeros((hps.hidden_size, T, bsize))
        hs = zeros((hps.hidden_size, T, bsize))
        dhs = zeros((hps.hidden_size, T, bsize))
        probs = zeros((hps.output_size, T, bsize))
        costs = np.zeros((T, bsize))

        if prev_h0 is not None:
            h0 = tile(prev_h0, bsize)
        else:
            h0 = tile(self.params['h0'], bsize)
        Wih = self.params['Wih']
        Whh = self.params['Whh']
        bhh = self.params['bhh']
        Who = self.params['Who']
        bho = self.params['bho']

        # Forward prop

        for t in xrange(T):
            if t == 0:
                hprev = h0
            else:
                hprev = hs[:, t-1, :]
            us[:, t, :] = mult(Wih, data[:, t, :]) + mult(Whh, hprev) + bhh
            hs[:, t, :] = self.nl(us[:, t, :])
            probs[:, t, :] = softmax(mult(Who, hs[:, t, :]) + bho)

        self.last_h = hs[:, -1, :].reshape((-1, 1))

        if labels is None:
            return None, probs

        probs_neg_log = as_np(-1 * log(probs))
        dprobs = as_np(probs.copy())
        for k in xrange(bsize):
            for t in xrange(len(labels[k])):
                costs[t, k] = probs_neg_log[labels[k][t], t, k]
                dprobs[labels[k][t], t, k] -= 1
        dprobs = array(dprobs)

        # NOTE Summing costs over time
        # NOTE FIXME Dividing by T to get better sense if objective
        # is decreasing, remove for grad checking
        cost = costs.sum() / bsize / float(T)
        if not back:
            return cost, probs

        # Backprop

        for k in self.grads:
            self.grads[k][:] = 0

        for t in reversed(xrange(T)):
            self.grads['bho'] += dprobs[:, t, :].sum(axis=-1).reshape((-1, 1)) / bsize
            self.grads['Who'] += mult(dprobs[:, t, :], hs[:, t, :].T) / bsize
            dhs[:, t, :] += mult(Who.T, dprobs[:, t, :])
            dus[:, t, :] += get_nl_grad(self.hps.nl, us[:, t, :]) * dhs[:, t, :]
            self.grads['Wih'] += mult(dus[:, t, :], data[:, t, :].T) / bsize
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

    dset = UttCharStream(args.batch_size)

    # Construct network
    model = RNN(dset, model_hps, opt_hps, opt='nag')
    model.run(check_grad=True)
