import cPickle as pickle
from ops import array, zeros, as_np, l2norm, square, sqrt
from optimizer import Optimizer
from log_utils import get_logger

'''
SGD w/ classical momentum
'''

logger = get_logger()

class MomentumOptimizer(Optimizer):

    '''
    max_grad: If set to float > 0, clips gradients
    rmsprop: If true, scales gradients by exponential weighted magnitudes
    '''

    def __init__(self, model, alpha=1e-3, mom=0.95, mom_low=0.5, low_mom_iters=100, max_grad=None, rmsprop=False, rmsprop_decay=0.99):
        super(MomentumOptimizer, self).__init__(model, alpha)
        # Momentum coefficient
        self.mom = mom
        self.mom_low = mom_low
        self.low_mom_iters = low_mom_iters
        self.max_grad = max_grad
        self.grad_norm = 0.0

        # Velocities
        self.vel = dict()
        if self.mom > 0:
            for p in self.params:
                self.vel[p] = zeros(self.params[p].shape)
            self.updates = self.vel
        else:
            self.vel = self.updates = dict()

        # Keep track of cost and smoothed cost
        self.costs = list()
        self.expcosts = list()

        self.rmsprop = rmsprop
        self.rmsprop_decay = rmsprop_decay
        if rmsprop:
            # Scale gradients by exponentially weighted average of magnitudes
            self.msgrads = dict()
            for p in self.params:
                self.msgrads[p] = None

    def get_mom(self):
        if self.iters < self.low_mom_iters:
            mom = self.mom_low
        else:
            mom = self.mom
        return mom

    def clip_grads(self, grads):
        # Gradient clipping
        self.grad_norm = 0.0
        for p in grads:
            self.grad_norm += l2norm(grads[p]) ** 2
        self.grad_norm = self.grad_norm ** 0.5
        if self.grad_norm > self.max_grad:
            logger.info('Clipping gradient by %f / %f' % (self.max_grad, self.grad_norm))
            return self.alpha * (self.max_grad / self.grad_norm)
        return self.alpha

    def rmsprop_update(self, grads):
        # NOTE In slides says momentum does not help as much
        # TODO Also for NAG they suggest dividing correction rather than vel
        if self.rmsprop:
            for p in self.msgrads:
                if self.msgrads[p] is None:
                    self.msgrads[p] = square(grads[p])
                else:
                    self.msgrads[p] = self.rmsprop_decay * self.msgrads[p] + (1-self.rmsprop_decay) * square(grads[p])

    def compute_update(self, data, labels):
        mom = self.get_mom()
        cost, grads = self.model.cost_and_grad(data, labels)
        self.update_costs(cost)

        self.rmsprop_update(grads)

        if self.max_grad is not None and self.max_grad > 0:
            alph = self.clip_grads(grads)
        else:
            alph = self.alpha

        for p in grads:
            if self.mom > 0:
                self.vel[p] = mom * self.vel[p] + alph * grads[p]
            else:
                # NOTE vel is updates
                self.vel[p] = alph * grads[p]

    def apply_update(self):
        for p in self.params:
            if self.rmsprop:
                # FIXME PARAM Smoothing parameter
                self.params[p] -= self.updates[p] / sqrt(self.msgrads[p] + 0.01)
            else:
                self.params[p] -= self.updates[p]

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
        if self.mom > 0:
            pickle.dump([as_np(self.vel[k]) for k in self.model.param_keys], fout)

    def from_file(self, fin):
        self.iters = pickle.load(fin)
        self.costs = pickle.load(fin)
        self.expcosts = pickle.load(fin)
        if self.mom > 0:
            loaded_vels = pickle.load(fin)
            # Put back on gpu
            self.vel = zip(self.model.param_keys, [array(v) for v in loaded_vels])
