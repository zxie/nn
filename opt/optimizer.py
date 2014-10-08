'''
Template class for optimizers

Given gradient for the current time step, computes
the update to apply (subtract) from each parameter
and applies update
'''

class Optimizer(object):

    def __init__(self, model, alpha=1e-3):
        self.alpha = alpha
        self.model = model
        self.params = model.params
        self.updates = dict()
        self.iters = 0

    def run(self, data, labels):
        # Function called externally
        cost = self.compute_update(data, labels)
        self.apply_update()
        self.iters += 1
        return cost

    def compute_update(self, grads):
        raise NotImplementedError()

    def to_file(self, fout):
        raise NotImplementedError()

    def from_file(self, fin):
        raise NotImplementedError()

    def apply_update(self):
        for p in self.params:
            self.params[p] -= self.updates[p]
