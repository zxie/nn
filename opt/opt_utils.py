from mom import MomentumOptimizer
from nag import NesterovOptimizer

def create_optimizer(name, model, alpha=1e-3, mom=0.95, mom_low=0.5, low_mom_iters=100):
    if name == 'cm':
        return MomentumOptimizer(model, alpha=alpha, mom=mom, mom_low=mom_low, low_mom_iters=low_mom_iters)
    elif name == 'nag':
        return NesterovOptimizer(model, alpha=alpha, mom=mom, mom_low=mom_low, low_mom_iters=low_mom_iters)
    else:
        assert False, 'No such optimizer %s' % name
