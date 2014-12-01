import numpy as np
import cPickle as pickle
from log_utils import get_logger
from ops import array, as_np, empty

# TODO Try and implement standard DNN, RNN, CNN models which
# other models can extend

logger = get_logger()

class Net(object):

    def __init__(self, dset, hps, train=True):
        self.params = dict()
        self.params_loaded = False
        self.dset = dset
        self.train = train
        self.hps = hps

    @staticmethod
    def init_hyperparams():
        raise NotImplementedError()

    def alloc_params(self):
        raise NotImplementedError()

    def cost_and_grad(self):
        raise NotImplementedError()

    def check_grad(self, data, labels, grads, eps=0.01, params_to_check=None):
        if not params_to_check:
            params_to_check = self.params.keys()
        for p in params_to_check:
            param = self.params[p]
            grad = grads[p]
            logger.info('Grad check on %s: %s' % (p, str(grad.shape)))
            # NOTE Want to use numpy at not 32 bit floats on GPU here
            num_grad = np.empty(param.shape, dtype=np.float64)
            for i in xrange(param.shape[0]):
                for j in range(param.shape[1]):
                    # NOTE Does 2-way numerical gradient
                    param[i, j] += eps
                    cost_p, _ = self.cost_and_grad(data, labels, back=False)
                    param[i, j] -= 2*eps
                    cost_m, _ = self.cost_and_grad(data, labels, back=False)
                    param[i, j] += eps
                    num_grad[i, j] = (cost_p - cost_m) / (2*eps)
                    print i, j, 'ng', num_grad[i, j], 'g', grad[i, j], '/',\
                        num_grad[i, j] / grad[i, j], '-', num_grad[i, j] - grad[i, j]

    def to_file(self, fout):
        logger.info('Saving state')
        pickle.dump([as_np(self.params[k]) for k in self.param_keys], fout)
        self.opt.to_file(fout)

    def from_file(self, fin):
        logger.info('Loading state')
        loaded_params = pickle.load(fin)
        self.params = dict(zip(self.param_keys, [array(param) for param in loaded_params]))
        if self.train:
            self.opt.from_file(fin)
        self.params_loaded = True

    def count_params(self):
        self.param_keys = sorted(self.params.keys())
        self.num_params = 0.0
        for k in self.param_keys:
            self.num_params += np.prod(self.params[k].shape)
        logger.info('Allocated %d parameters' % self.num_params)

    def alloc_grads(self):
        # Call after allocating parameters
        self.grads = {}
        for k in self.params:
            self.grads[k] = empty(self.params[k].shape)
        logger.info('Allocated gradients')

    def update_params(self, data, labels):
        self.opt.run(data, labels)

    def run(self, back=True):
        if not back and not self.params_loaded:
            logger.warn('Running testing without having loaded parameters')

    def start_next_epoch(self):
        self.dset.restart(shuffle=True)
