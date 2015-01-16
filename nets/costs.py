import numpy as np
import ops
from ops import square, absval, array, as_np

def l2_cost(X, Y):
    costs = square(X - Y)
    deltas = 2 * (X - Y)
    return costs, deltas

def l1_cost(X, Y):
    costs = absval(Y - X)
    deltas = -1 * array(np.sign(as_np(Y - X)))
    return costs, deltas

def cross_ent_cost(X, y):
    pass
