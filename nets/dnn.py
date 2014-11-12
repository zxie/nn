import numpy as np
from log_utils import get_logger
from ops import get_nl, vp_init, zeros, mult, empty, get_nl_grad,\
        softmax, as_np, array
from models import Net
from param_utils import ParamStruct, ModelHyperparams
from optimizer import OptimizerHyperparams
from opt_utils import create_optimizer
from char_corpus import CharCorpus, CONTEXT
from dset_utils import one_hot

logger = get_logger()

class DNNHyperparams(ModelHyperparams):

    def __init__(self, **entries):
        # TODO Don't assume hidden layer sizes are the same
        self.defaults = [
            ('hidden_size', 1200, 'size of hidden layers'),
            ('hidden_layers', 2, 'number of hidden layers'),
            # TODO Determine this from dataset input
            ('input_size', 34*CONTEXT, 'input dimension size'),
            ('output_size', 34, 'size of softmax output'),
            ('batch_size', 512, 'size of dataset batches'),
            ('nl', 'relu', 'type of nonlinearity')
        ]
        super(DNNHyperparams, self).__init__(entries)

class DNN(Net):

    def __init__(self, dset, hps, opt_hps, train=True, opt='nag'):

        super(DNN, self).__init__(dset, hps, train=train)
        self.nl = get_nl(hps.nl)

        self.alloc_params()

        if train:
            self.opt = create_optimizer(opt, self, alpha=opt_hps.alpha,
                    mom=opt_hps.mom, mom_low=opt_hps.mom_low,
                    low_mom_iters=opt_hps.low_mom_iters)

    @staticmethod
    def init_hyperparams():
        return DNNHyperparams()

    def alloc_params(self):
        hps = self.hps

        self.params['Wih'] = vp_init((hps.hidden_size, hps.input_size))
        self.params['bih'] = zeros((hps.hidden_size, 1))

        for k in xrange(hps.hidden_layers - 1):
            self.params['W%d' % (k+1)] = vp_init((hps.hidden_size, hps.hidden_size))
            self.params['b%d' % (k+1)] = zeros((hps.hidden_size, 1))

        self.params['Who'] = vp_init((hps.output_size, hps.hidden_size))
        self.params['bho'] = zeros((hps.output_size, 1))

        self.count_params()

        # Allocate grads as well

        self.grads = {}
        for k in self.params:
            self.grads[k] = empty(self.params[k].shape)
        logger.info('Allocated gradients')

    def run(self, back=True):
        super(DNN, self).run(back=back)

        data, labels = self.dset.get_batch()
        data = one_hot(data, self.hps.output_size)
        data = data.reshape((-1, data.shape[2]))
        #cost, grads = self.cost_and_grad(data, labels)
        #self.check_grad(data, labels, grads, params_to_check=['Wih'], eps=0.01)
        if back:
            self.update_params(data, labels)
        else:
            cost, probs = self.cost_and_grad(data, labels, back=False)
            return cost, probs

    def cost_and_grad(self, data, labels, back=True):
        hps = self.hps
        grads = self.grads

        # May not be full batch size if at end of dataset
        bsize = data.shape[-1]

        p = ParamStruct(**self.params)

        # Forward prop

        acts = list()
        acts.append(self.nl(mult(p.Wih, data) + p.bih))

        for k in xrange(hps.hidden_layers - 1):
            W = self.params['W%d' % (k+1)]
            b = self.params['b%d' % (k+1)]
            acts.append(self.nl(mult(W, acts[-1]) + b))

        y = mult(p.Who, acts[-1]) + p.bho
        probs = softmax(y)

        if labels is None:
            return None, probs

        # NOTE For more precision if necessary convert to nparray early
        cost_array = np.empty(bsize, dtype=np.float64)
        # Speed things up by doing assignments off gpu
        neg_log_prob = -1 * np.log(as_np(probs))
        for k in xrange(bsize):
            cost_array[k] = neg_log_prob[labels[k], k]
        cost = cost_array.sum() / bsize

        if not back:
            return cost, probs

        # Backprop

        for k in self.grads:
            self.grads[k][:] = 0

        # Do assignments off GPU to speed things up
        dLdy = as_np(probs)
        # NOTE This changes probs
        for k in xrange(bsize):
            dLdy[labels[k], k] -= 1
        dLdy = array(dLdy)

        grads['bho'] = dLdy.sum(axis=1).reshape((-1, 1))
        grads['Who'] = mult(dLdy, acts[-1].T)
        Ws = [p.Wih] + [self.params['W%d' % (k+1)] for k in xrange(hps.hidden_layers - 1)] + [p.Who]
        deltas = [dLdy]

        for k in reversed(xrange(hps.hidden_layers - 1)):
            delta = get_nl_grad(self.hps.nl, acts[k+1]) * mult(Ws[k + 2].T, deltas[-1])
            deltas.append(delta)
            grads['b%d' % (k+1)] = delta.sum(axis=1).reshape((-1, 1))
            grads['W%d' % (k+1)] = mult(delta, acts[k].T)

        delta = get_nl_grad(self.hps.nl, acts[0]) * mult(Ws[1].T, deltas[-1])
        grads['bih'] = delta.sum(axis=1).reshape((-1, 1))
        grads['Wih'] = mult(delta, data.T)

        # Normalize
        for k in self.grads:
            self.grads[k] /= bsize

        return cost, self.grads


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    model_hps = DNNHyperparams()
    opt_hps = OptimizerHyperparams()
    model_hps.add_to_argparser(parser)
    opt_hps.add_to_argparser(parser)

    args = parser.parse_args()

    model_hps.set_from_args(args)
    opt_hps.set_from_args(args)

    from dset_paths import SWBD_CORPUS_DATA_FILE
    dset = CharCorpus(CONTEXT, args.batch_size, data_file=SWBD_CORPUS_DATA_FILE)

    # Construct network
    model = DNN(dset, model_hps, opt_hps, opt='nag')
    model.run()
