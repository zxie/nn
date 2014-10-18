from param_utils import ModelHyperparams
from nplm import NPLM
from log_utils import get_logger

'''
Same as NPLM except with characters, some different
parameter settings
'''

logger = get_logger()

class NCLMHyperparams(ModelHyperparams):

    def __init__(self, **entries):
        self.defaults = [
            ('embed_size', 10, 'size of char embeddings'),
            ('context_size', 11, 'size of char context (so 7 for 8-gram)'),
            ('hidden_size', 1000, 'size of hidden layer'),
            ('batch_size', 512, 'size of dataset batches'),
            # Not really a hyperparameter...
            ('nl', 'relu', 'type of nonlinearity')
        ]
        super(NCLMHyperparams, self).__init__(entries)

class NCLM(NPLM):
    pass
