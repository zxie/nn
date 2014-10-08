import cPickle as pickle
from ops import array
from ops import zeros
from optimizer import Optimizer

'''
SGD w/ classical momentum
'''


class MomentumOptimizer(Optimizer):

    def __init__(self, model, alpha=1e-3, mom=0.95, mom_low=0.5, low_mom_iters=100):
        super(MomentumOptimizer, self).__init__(model, alpha)
        # Momentum coefficient
        self.mom = mom
        self.mom_low = mom_low
        self.low_mom_iters = low_mom_iters

        # Velocities
        self.vel = dict()
        for p in self.params:
            self.vel[p] = zeros(self.params[p].shape)
        self.updates = self.vel

    def get_mom(self):
        if self.iters < self.low_mom_iters:
            mom = self.mom_low
        else:
            mom = self.mom
        return mom

    def compute_update(self, data, labels):
        cost, grads = self.model.cost_and_grad(data, labels)
        mom = self.get_mom()
        for p in grads:
            self.vel[p] = mom * self.vel[p] + self.alpha * grads[p]
        return cost

    def to_file(self, fout):
        pickle.dump(self.iters, fout)
        pickle.dump([self.vel[k].as_numpy_array() for k in self.model.param_keys], fout)

    def from_file(self, fin):
        self.iters = pickle.load(fin)
        loaded_vels = pickle.load(fin)
        self.vel = zip(self.model.param_keys(), [array(v) for v in loaded_vels])
