from param_utils import HyperparamStruct

'''
Template class and hyperparameters for optimizers

Given gradient for the current time step, computes
the update to apply (subtract) from each parameter
and applies update
'''


class OptimizerHyperparams(HyperparamStruct):

    def __init__(self, **entries):
        self.defaults = [
            ('alpha', 0.01, 'step size'),
            ('mom', 0.95, 'momentum coefficient (after low_mom_iters)'),
            ('mom_low', 0.5, 'low momentum prior to low_mom_iters'),
            ('low_mom_iters', 100, 'number of iterations to run with low momentum'),
            ('max_grad', -1.0, 'clip gradients'),
            ('rmsprop', False, 'scale gradients by exponentially weighted average of magnitudes'),
            ('rmsprop_decay', 0.99, 'exponential decay when using rmsprop')
        ]

        super(OptimizerHyperparams, self).__init__(entries)


class Optimizer(object):

    def __init__(self, model, alpha=1e-3):
        self.alpha = alpha
        self.model = model
        self.params = model.params
        self.updates = dict()
        self.iters = 0

    def run(self, data, labels):
        # Function called externally
        self.compute_update(data, labels)
        self.apply_update()
        self.iters += 1

    def compute_update(self, grads):
        raise NotImplementedError()

    def to_file(self, fout):
        raise NotImplementedError()

    def from_file(self, fin):
        raise NotImplementedError()

    def apply_update(self):
        for p in self.params:
            self.params[p] -= self.updates[p]
