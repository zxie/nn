import numpy as np
from ops import mult, tanh, sigmoid, relu, array, rand, zeros, ones, empty
from log_utils import get_logger

# NOTE Can get speed savings combining softmax with -logprob node
# since have simplified closed form for combined gradient

logger = get_logger()

class Node(object):

    def __init__(self, name):
        self.name = name
        self.succ = list()
        self.pred = list()
        self.grad = None
        self.full_grad = None

    def __str__(self):
        return '%s [%s]' % (self.__class__.__name__, self.name)

    def fprop(self):
        pass

    def compose_grad(self):
        # NOTE Handles one node having multiple successors
        assert (self.grad is not None)
        print 'succ grad shapes:', [(str(s), s.full_grad.shape) for s in self.succ]
        print 'self grad shape:', (str(self), self.grad.shape)
        if hasattr(self.succ, '__len__'):
            self.full_grad = self.grad * sum([s.full_grad for s in self.succ])
        else:
            self.full_grad = self.grad * self.succ.full_grad

class ParamNode(Node):

    def __init__(self, name, shape, init_fn=None):
        super(ParamNode, self).__init__(name)
        self.shape = shape
        if init_fn is None:
            self.params = rand(shape)
        else:
            self.params = init_fn(shape)
        self.out = self.params

    def shape(self):
        # NOTE Returns full shape for IndexedParamNode
        return self.params.shape

class IndexedParamNode(ParamNode):

    '''
    Given a set of indices, propagates the parameter
    vectors associated with those indices (e.g. for word embeddings)
    '''

    # NOTE Could also do this using one-hot vectors and a mat-mult

    def __init__(self, name, data_inp, shape, init_fn=None):
        super(IndexedParamNode, self).__init__(name, shape, init_fn=init_fn)
        self.data_inp = data_inp
        self.params_batch = empty((data_inp.feat_dim * self.params.shape[0],
            data_inp.batch_size))

    def fprop(self):
        data_batch, labels = self.data_inp.get_batch()
        logger.debug('%s prop: data shape %s, inds shape %s' % (str(self), str(self.params.shape), str(data_batch.shape)))
        # Flatten/stack together the vectors in each col of the batch and put on the GPU
        # TODO Speed up this portion
        for k in range(data_batch.shape[1]):
            self.params_batch[:, k] = self.params[:, data_batch[:, k]].ravel()
        self.out = self.params_batch

    def bprop(self):
        # TODO
        pass

class DataNode(Node):
    # TODO Datasets should have these types of nodes for
    # producing their batches
    pass

class CallNode(Node):
    '''
    Abstract class which takes input, applies one computation, and
    then passes to output
    '''

    def __init__(self, name, pred):
        super(CallNode, self).__init__(name)
        if hasattr(pred, '__iter__'):
            for n in pred:
                n.succ.append(self)
        else:
            pred.succ.append(self)
        self.pred = pred

    def bprop(self):
        raise NotImplementedError()

class SumNode(CallNode):

    def __init__(self, name, pred):
        super(SumNode, self).__init__(name, pred)

    def fprop(self):
        self.out = sum([n.out for n in self.pred])

    def bprop(self):
        logger.debug('%s backprop' % str(self))
        # TODO This can be merged / sped up
        self.grad = ones(self.pred[0].out.shape)
        self.full_grad = self.grad

class LinearNode(CallNode):
    '''
    Represents operation f(x) = W*x
    '''

    def __init__(self, name, pred):
        super(LinearNode, self).__init__(name, pred)
        self.x = pred[0]
        self.W = pred[1]

    def fprop(self):
        self.out = mult(self.W.out, self.x.out)

    def bprop(self):
        logger.debug('%s backprop' % str(self))
        # FIXME Assuming just 1 successor for now
        assert len(self.succ) == 1
        succ_grad = self.succ[0].full_grad
        print self.W.shape
        self.full_grad = self.W.out.T * succ_grad
        # TODO Check this
        self.W.grad = mult(succ_grad, self.succ[0].out.T)

class AffineNode(CallNode):
    '''
    Represents operation f(x) = W*x + b
    '''

    def __init__(self, name, pred):
        super(AffineNode, self).__init__(name, pred)
        self.x = pred[0]
        self.W = pred[1]
        self.b = pred[2]

    def fprop(self):
        logger.debug('%s prop: %s x %s + %s' % (str(self), str(self.W.shape), str(self.x.out.shape), str(self.b.out.shape)))
        self.out = mult(self.W.out, self.x.out) + self.b.out

    def bprop(self):
        logger.debug('%s backprop' % str(self))
        # FIXME Assuming just 1 successor for now
        assert len(self.succ) == 1
        succ_grad = self.succ[0].full_grad

        # FIXME Hack to avoid large sparse matrix multiply
        if type(self.succ[0]) is SumNode:  # Which leads to softmax...
            if self.full_grad is None:
                self.full_grad = empty((self.W.shape[1], succ_grad.shape[1]))
                self.W.grad = empty(self.W.shape)

            for k in range(self.full_grad.shape[0]):
                # TODO self.full_grad[k, :] = ?
                # TODO self.W.grad[k, :] = ?
                pass
        else:
            self.full_grad = mult(self.W.out.T, succ_grad)
            # FIXME Multiplication below is wrong
            self.W.grad = mult(succ_grad, self.succ[0].out.T)

        # TODO Check this
        self.b.grad = succ_grad

class ReluNode(CallNode):

    def fprop(self):
        self.out = relu(self.pred.out)

    def bprop(self):
        logger.debug('%s backprop' % str(self))
        self.grad = (self.out > 0)
        self.compose_grad()

class SigmoidNode(CallNode):

    def fprop(self):
        self.out = sigmoid(self.pred.out)

    def bprop(self):
        logger.debug('%s backprop' % str(self))
        self.grad = self.out * (1 - self.out)
        self.compose_grad()

class TanhNode(CallNode):

    def fprop(self):
        logger.debug('%s prop: %s' % (str(self), str(self.pred.out.shape)))
        self.out = tanh(self.pred.out)

    def bprop(self):
        logger.debug('%s backprop' % str(self))
        self.grad = (1 - self.out * self.out)
        self.compose_grad()

# TODO Look for ways to speed this up since the softmax gradients
# are extremely sparse -- 1 non-zero over the classes * # elements in batch

class SoftmaxNode(CallNode):

    def fprop(self):
        logger.debug('%s prop: %s' % (str(self), str(self.pred.out.shape)))
        # First subtract away the maximum value and exponentiate
        probs = (self.pred.out - self.pred.out.max(axis=0)).exp()
        # Normalize
        probs = probs / probs.sum(axis=0)
        self.out = probs

    def bprop(self):
        logger.debug('%s backprop' % str(self))
        self.grad = (self.out*self.out - self.out)
        self.compose_grad()

class ObjectiveNode(CallNode):

    # FIXME TODO Make this more general and split into multiple classes

    def __init__(self, name, pred, labels_func):
        super(ObjectiveNode, self).__init__(name, pred)
        # FIXME Ugly way of not repeatedly fetching batches
        self.labels_func = labels_func

    def fprop(self):
        logger.debug('%s prop' % str(self))
        self.labels = self.labels_func()
        batch_size = self.labels.size
        # Sum negative log probabilities
        self.out = -1.0/batch_size * self.pred.out[self.labels].log().sum()

    def bprop(self):
        logger.debug('%s backprop' % str(self))
        logger.debug('labels shape: %s' % str(self.labels.shape))
        # NOTE Assumes ObjectiveNode has no successors
        batch_size = self.labels.size
        self.full_grad = zeros(self.pred.out.shape)
        for k in range(self.labels.size):
            self.full_grad[self.labels[k], k] = -1.0/batch_size * (1 / self.pred.out[self.labels[k], k])
