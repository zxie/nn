import matplotlib.pyplot as plt
from numpy import array, zeros, exp, dot
from numpy import meshgrid, arange
from numpy.random import rand, randn


# size of bounding box: SIZE X SIZE.
SIZE = 10
RADIUS_RATIO = 1.2


def norm(x):
    return ((x ** 2).sum()) ** 0.5


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


def bounce_n(T=128, n=2, r=None, m=None):
    if r is None:
        r = array([RADIUS_RATIO] * n)
    if m is None:
        m = array([1] * n)
    # r is to be rather small.
    X = zeros((T, n, 2))
    v = randn(n, 2)
    v = v / norm(v) * .5
    good_config = False
    while not good_config:
        x = 2 + rand(n, 2) * 8
        good_config = True
        for i in range(n):
            for z in range(2):
                if x[i][z] - r[i] < 0:
                    good_config = False
                if x[i][z] + r[i] > SIZE:
                    good_config = False
# that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i] - x[j]) < r[i] + r[j]:
                    good_config = False

    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        for i in range(n):
            X[t, i] = x[i]

        for mu in range(int(1 / eps)):

            for i in range(n):
                x[i] += eps * v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z] - r[i] < 0:
                        v[i][z] = abs(v[i][z])  # want positive
                    if x[i][z] + r[i] > SIZE:
                        v[i][z] = -abs(v[i][z])  # want negative

            for i in range(n):
                for j in range(i):
                    if norm(x[i] - x[j]) < r[i] + r[j]:
                        # the bouncing off part:
                        w = x[i] - x[j]
                        w = w / norm(w)

                        v_i = dot(w.transpose(), v[i])
                        v_j = dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i] += w * (new_v_i - v_i)
                        v[j] += w * (new_v_j - v_j)

    return X


def ar(x, y, z):
    return z / 2 + arange(x, y, z, dtype='float')


def matricize(X, res, r=None):

    T, n = X.shape[0:2]
    if r is None:
        r = array([RADIUS_RATIO] * n)

    A = zeros((T, res, res))

    [I, J] = meshgrid(ar(0, 1, 1. / res) * SIZE, ar(0, 1, 1. / res) * SIZE)

    for t in range(T):
        for i in range(n):
            A[t] += exp(-(((I - X[t, i, 0]) ** 2 + (J - X[t, i, 1])
                           ** 2) / (r[i] ** 2)) ** 4)

        A[t][A[t] > 1] = 1
    return A


def bounce_mat(res, n=2, T=128, r=None):
    if r is None:
        r = array([RADIUS_RATIO] * n)
    x = bounce_n(T, n, r)
    A = matricize(x, res, r)
    return A


def bounce_vec(res, n=2, T=128, r=None, m=None, ret_coords=False):
    if r is None:
        r = array([RADIUS_RATIO] * n)
    x = bounce_n(T, n, r, m)
    V = matricize(x, res, r)
    V = V.reshape(T, res ** 2)
    if ret_coords:
        return V, x
    else:
        return V


def show_single_V(V):
    res = int(V.shape[0] ** 0.5)
    show(V.reshape(res, res))


def show_V(V):
    T = V.shape[0]
    res = int(V.shape[1] ** 0.5)
    for t in range(T):
        plt.imshow(V[t].reshape(res, res))
        plt.show()


def show_A(A):
    T = len(A)
    for t in range(T):
        show(A[t])
