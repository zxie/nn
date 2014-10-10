import gnumpy as gnp

'''
Ideally hide away all low-level libraries so easier to swap one library
out with another for later optimization

Currently using gnumpy on top of cudamat
'''

# Convert numpy / other array to type we'll use

def array(arr):
    return gnp.garray(arr)

def empty(shape):
    return gnp.empty(shape)

def rand(shape, rg=[0, 1]):
    return gnp.rand(shape) * (rg[1] - rg[0]) + rg[0]

def zeros(shape):
    return gnp.zeros(shape)

def ones(shape):
    return gnp.ones(shape)

# Nonlinearities

def relu(x):
    return x * (x > 0)

def sigmoid(x):
    return x.logistic()

def tanh(x):
    return x.tanh()

def get_nl(nl):
    if nl == 'relu':
        return relu
    elif nl == 'sigmoid':
        return sigmoid
    elif nl == 'tanh':
        return tanh
    else:
        assert False, 'No such nonlinearity: %s' % nl

def get_nl_grad(nl, act):
    if nl == 'relu':
        return act > 0
    elif nl == 'sigmoid':
        return act * (1 - act)
    elif nl == 'tanh':
        return (1 - act*act)
    else:
        assert False, 'No such nonlinearity: %s' % nl


# TODO RelU

# Matrix multiply

def mult(A, B):
    return gnp.dot(A, B)
