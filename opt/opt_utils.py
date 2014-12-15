from mom import MomentumOptimizer
from nag import NesterovOptimizer

def create_optimizer(name, model, **kwargs):
    if name == 'cm':
        return MomentumOptimizer(model, **kwargs)
    elif name == 'nag':
        return NesterovOptimizer(model, **kwargs)
    else:
        assert False, 'No such optimizer %s' % name
