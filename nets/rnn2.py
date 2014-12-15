import numpy as np
from optimizer import OptimizerHyperparams
from ops import zeros, get_nl, softmax, mult,\
        get_nl_grad, as_np, array, log, vp_init,\
        USE_GPU, gnp, empty, tile, rand
from models import Net
from log_utils import get_logger
from param_utils import ModelHyperparams
from utt_char_stream import UttCharStream
from opt_utils import create_optimizer
from dset_utils import one_hot_lists

# NOTE Uncomment this for grad-checking
#def vp_init(shape):
    #return rand(shape, (-0.01, 0.01))

# TODO
# - Bi-directional
# - mRNN (hopefully can just subclass and make few changes)
# - Need to figure out best nonlinearities too

logger = get_logger()

class RNNHyperparams(ModelHyperparams):

    def __init__(self, **entries):
        self.defaults = [
            ('hidden_size', 1000, 'size of hidden layers'),
            ('hidden_layers', 5, 'number of hidden layers'),
            ('recurrent_layer', 3, 'layer which should have recurrent connections'),
            ('output_size', 34, 'size of softmax output'),
            ('batch_size', 128, 'size of dataset batches'),
            ('max_act', 5.0, 'threshold to clip activation'),
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
            self.opt = create_optimizer(opt, self, **(opt_hps.to_dict()))

    @staticmethod
    def init_hyperparams():
        return RNNHyperparams()

    def alloc_params(self):
        # Refer to Ch. 2 pg. 10 of Sutskever's thesis

        hps = self.hps

        # Initial hidden state
        self.params['h0'] = zeros((hps.hidden_size, hps.hidden_layers))

        # Input to hidden, note if first layer is recurrent bih is redundant
        self.params['Wih'] = vp_init((hps.hidden_size, hps.output_size))
        self.params['bih'] = zeros((hps.hidden_size, 1))

        # recurrent weight
        # NOTE Initialization important for grad check, don't use vp_init?
        self.params['Whh'] = vp_init((hps.hidden_size, hps.hidden_size))
        self.params['bhh'] = zeros((hps.hidden_size, 1))

        # Weights between hidden layers
        for k in xrange(1, hps.hidden_layers):
            self.params['Wh%d' % k] = vp_init((hps.hidden_size, hps.hidden_size))
            self.params['bh%d' % k] = zeros((hps.hidden_size, 1))

        # Hidden to output
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
            self.check_grad(data, labels, grads, params_to_check=['Wh1'], eps=0.1)
        else:
            if back:
                self.update_params(data, labels)
            else:
                cost, probs = self.cost_and_grad(data, labels, back=False)
                return cost, probs

    def cost_and_grad(self, data, labels, back=True, prev_h0=None):
        hps = self.hps
        T = data.shape[1]
        bsize = data.shape[2]

        # FIXME gnumpy reallocates if try and use same parameters?
        #us = self.us[:, 0:T, 0:bsize]
        #dus = self.dus[:, 0:T, 0:bsize]
        #hs = self.hs[:, 0:T, 0:bsize]
        #dhs = self.dhs[:, 0:T, 0:bsize]
        #probs = self.probs[:, 0:T, 0:bsize]
        #dprobs = self.dprobs[:, 0:T, 0:bsize]
        #costs = self.costs[0:T, 0:bsize]

        us = list()
        dus = list()
        hs = list()
        dhs = list()
        h0 = list()
        for k in xrange(hps.hidden_layers):
            us.append(zeros((hps.hidden_size, T, bsize)))
            dus.append(zeros((hps.hidden_size, T, bsize)))
            hs.append(zeros((hps.hidden_size, T, bsize)))
            dhs.append(zeros((hps.hidden_size, T, bsize)))
            h0.append(empty((hps.hidden_size, bsize)))
        probs = zeros((hps.output_size, T, bsize))
        costs = np.zeros((T, bsize))
        if prev_h0 is not None:
            h0 = prev_h0
        else:
            for k in xrange(hps.hidden_layers):
                h0[k] = tile(self.params['h0'][:, k].reshape(-1, 1), bsize)
        bih = self.params['bih']
        Wih = self.params['Wih']
        Whh = self.params['Whh']
        bhh = self.params['bhh']
        Who = self.params['Who']
        bho = self.params['bho']

        # Forward prop

        for t in xrange(T):
            for k in xrange(hps.hidden_layers):
                if t == 0:
                    hprev = h0[k]
                else:
                    hprev = hs[k][:, t-1, :]

                if k == 0:
                    us[k][:, t, :] = mult(Wih, data[:, t, :]) + bih
                else:
                    us[k][:, t, :] = mult(self.params['Wh%d' % k], hs[k-1][:, t, :])

                if k == hps.recurrent_layer - 1:
                    us[k][:, t, :] += mult(Whh, hprev) + bhh
                    # Clip maximum activation
                    mask = us[k][:, t, :] < hps.max_act
                    us[k][:, t, :] = us[k][:, t, :] * mask + hps.max_act * (1 - mask)
                elif k != 0:
                    us[k][:, t, :] += self.params['bh%d' % k]

                hs[k][:, t, :] = self.nl(us[k][:, t, :])

            probs[:, t, :] = softmax(mult(Who, hs[-1][:, t, :]) + bho)

        self.last_h = hs[-1][:, :, :]

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
            self.grads['Who'] += mult(dprobs[:, t, :], hs[-1][:, t, :].T) / bsize

            for k in reversed(xrange(hps.hidden_layers)):
                if k == hps.hidden_layers - 1:
                    dhs[k][:, t, :] += mult(Who.T, dprobs[:, t, :])
                else:
                    dhs[k][:, t, :] += mult(self.params['Wh%d' % (k+1)].T, dhs[k+1][:, t, :])
                dus[k][:, t, :] += get_nl_grad(self.hps.nl, us[k][:, t, :]) * dhs[k][:, t, :]

                if k > 0:
                    self.grads['Wh%d' % k] += mult(dus[k][:, t, :], hs[k-1][:, t, :].T) / bsize
                    self.grads['bh%d' % k] += dus[k][:, t, :].sum(axis=-1).reshape((-1, 1)) / bsize

                if k == hps.recurrent_layer - 1:
                    if t == 0:
                        hprev = h0[k]
                        self.grads['h0'][:, k] = mult(Whh.T, dus[k][:, t, :]).sum(axis=-1) / bsize
                    else:
                        hprev = hs[k][:, t-1, :]
                        dhs[k][:, t-1, :] = mult(Whh.T, dus[k][:, t, :]).sum(axis=-1).reshape(-1, 1) / bsize
                    self.grads['Whh'] += mult(dus[k][:, t, :], hprev.T) / bsize
                    self.grads['bhh'] += dus[k][:, t, :].sum(axis=-1).reshape((-1, 1)) / bsize

            self.grads['Wih'] += mult(dus[0][:, t, :], data[:, t, :].T) / bsize
            self.grads['bih'] += dus[0][:, t, :].sum(axis=-1).reshape((-1, 1)) / bsize

        return cost, self.grads


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    model_hps = RNNHyperparams()
    model_hps.hidden_size = 10
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
