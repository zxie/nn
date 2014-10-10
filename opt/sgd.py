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

        # Keep track of cost and smoothed cost
        self.costs = list()
        self.expcosts = list()

    def get_mom(self):
        if self.iters < self.low_mom_iters:
            mom = self.mom_low
        else:
            mom = self.mom
        return mom

    def compute_update(self, data, labels):
        mom = self.get_mom()
        cost, grads = self.model.cost_and_grad(data, labels)
        self.update_costs(cost)
        for p in grads:
            self.vel[p] = mom * self.vel[p] + self.alpha * grads[p]

    def update_costs(self, cost):
        self.costs.append(cost)
        if not self.expcosts:
            self.expcosts.append(cost)
        else:
            # PARAM
            self.expcosts.append(0.01*cost + 0.99*self.expcosts[-1])

    def to_file(self, fout):
        pickle.dump(self.iters, fout)
        pickle.dump(self.costs, fout)
        pickle.dump(self.expcosts, fout)
        pickle.dump([self.vel[k].as_numpy_array() for k in self.model.param_keys], fout)

    def from_file(self, fin):
        self.iters = pickle.load(fin)
        self.costs = pickle.load(fin)
        self.expcosts = pickle.load(fin)
        loaded_vels = pickle.load(fin)
        # Put back on gpu
        self.vel = zip(self.model.param_keys, [array(v) for v in loaded_vels])
