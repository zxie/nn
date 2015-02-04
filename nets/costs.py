import numpy as np
from ops import square, absval, array, as_np, get_nl_grad, copy_arr

def l2_cost(X, Y):
    # Corresponds to output of a = W*x + b
    costs = square(X - Y)
    deltas = 2 * (X - Y)
    return costs, deltas


def unit_l2_cost(X, Y):
    # Corresponds to output of a = sigmoid(W*x + b)
    costs = square(X - Y)
    deltas = 2 * (X - Y) * get_nl_grad('sigmoid', X)
    return costs, deltas


def l1_cost(X, Y):
    # Corresponds to output of a = W*x + b
    costs = absval(Y - X)
    deltas = -1 * array(np.sign(as_np(Y - X)))
    return costs, deltas


def cross_ent_cost(probs, labels):
    deltas = as_np(copy_arr(probs))
    probs_neg_log = -1 * np.log(deltas)
    costs = np.zeros(probs.shape[1])

    bsize = len(labels)
    for k in xrange(bsize):
        for t in xrange(len(labels[k])):
            # NOTE Very slow if probs_neg_log not in CPU memory
            costs[t*bsize + k] = probs_neg_log[labels[k][t], t*bsize+k]
            deltas[labels[k][t], t*bsize+k] -= 1

    return costs, array(deltas)
