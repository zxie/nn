import cPickle as pickle
from log_utils import get_logger
from ops import array

logger = get_logger()

class Net(object):

    def __init__(self, dset, opt='nag'):
        raise NotImplementedError()

    def alloc_params(self):
        raise NotImplementedError()

    def cost_and_grad(self):
        raise NotImplementedError()

    def check_grad(self):
        raise NotImplementedError()

    def to_file(self, fout):
        logger.info('Saving state')
        # TODO Move this to parent model class
        self.opt.to_file(fout)
        pickle.dump([self.params[k].as_numpy_array() for k in self.param_keys], fout)

    def from_file(self, fin):
        logger.info('Loading state')
        self.opt.from_file(fin)
        loaded_params = pickle.load(fin)
        self.params = dict(zip(self.param_keys, [array(param) for param in loaded_params]))

    def update_params(self, data, labels):
        cost = self.opt.run(data, labels)
        return cost
