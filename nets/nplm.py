import numpy as np
from ops import rand, zeros
from graph import topological_traverse, get_all_nodes
from nodes import ParamNode, IndexedParamNode, AffineNode, TanhNode, SumNode,\
        ObjectiveNode, SoftmaxNode, LinearNode
from dset import BrownCorpus
from nag import NesterovOptimizer
from log_utils import get_logger

'''
Implementation of
    "A Neural Probabilistic Language Model",
    Bengio et. al., JMLR 2003
Follows some details given in
    "Decoding with Large-Scale Neural LMs Improves Translation",
    Vaswani et. al., EMNLP 2013
'''

logger = get_logger()

# TODO Handle hyperparameters some other way
embed_size = 30  # size of word embeddings
context_size = 4  # size of word context (so 4 for 5-gram)
hidden_size = 100
batch_size = 512
rand_range = [0.01, 0.01]

# TODO Parent class Net
class NPLM(object):

    def __init__(self, opt, dset):
        self.opt = opt
        self.dset = dset
        self.vocab_size = len(dset.word_inds)
        logger.debug('Vocab size: %d' % self.vocab_size)

        self.alloc_params()
        self.build_graph()

        self.root = self.C

    def alloc_params(self):
        rand_init = lambda shape: rand(shape, rand_range)
        # PARAM Following Vaswani et. al. EMNLP 2013
        bias_init = lambda shape: zeros(shape) - np.log(self.vocab_size)
        # NOTE IndexedParamNode allocates batch of values indexed from C
        self.C = IndexedParamNode('x = C[:, ks]', self.dset, (embed_size, self.vocab_size), init_fn=rand_init)
        self.H = ParamNode('H', (hidden_size, context_size*embed_size), init_fn=rand_init)
        self.d = ParamNode('d', (hidden_size, 1), init_fn=bias_init)
        self.U = ParamNode('U', (self.vocab_size, hidden_size), init_fn=rand_init)
        self.b = ParamNode('b', (self.vocab_size, 1), init_fn=bias_init)
        self.W = ParamNode('W', (self.vocab_size, context_size*embed_size), init_fn=rand_init)
        self.param_nodes = [self.C, self.H, self.d, self.U, self.b, self.W]
        logger.info('Allocated parameters')

    def build_graph(self):
        o = AffineNode('o = H*x + d', [self.C, self.H, self.d])
        a = TanhNode('a = tanh(o)', o)
        # TODO Could parallelize things here
        y = AffineNode('y = U*a + b', [a, self.U, self.b])
        y = SumNode('y = y + W*x', [y, LinearNode('W*x', [self.C, self.W])])
        self.softmax = SoftmaxNode('softmax', y)
        self.obj = ObjectiveNode('acc', self.softmax, self.get_labels)
        self.nodes = get_all_nodes(self.param_nodes)
        logger.info('Built graph')

    # FIXME Better way to handle this
    def get_labels(self):
        return self.dset.batch_labels

    def run(self):
        topological_traverse(self.nodes, back=True)


if __name__ == '__main__':
    # TODO Claim GPU
    # TODO Split into train and test

    # Load dataset
    brown = BrownCorpus(context_size, batch_size)
    nag = NesterovOptimizer()

    # TODO Create optimizer

    # Construct network
    nplm = NPLM(nag, brown)
    obj = nplm.obj

    # Run training
    epochs = 1
    for k in xrange(0, epochs):
        brown.restart()
        it = 0
        while brown.data_left():
            nplm.run()
            logger.info('epoch %d, iter %d, obj=%f' % (k, it, obj.out))
            it += 1
