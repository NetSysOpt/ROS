import random
from scipy.sparse import csr_matrix, linalg
import numpy as np
import logging
import time

def bregman_kl(X, Y):
    X = np.where(X == 0, np.finfo(float).eps, X)
    Y = np.where(Y == 0, np.finfo(float).eps, Y)
    return np.sum(X * np.log(X / Y))


def get_matrix(args, nx_G):
    ls_a = []
    ls_b = []
    ls_inter = []
    for (u, v, val) in nx_G.edges(data = True):
        ls_a.append(u)
        ls_b.append(v)
        ls_inter.append(val['weight'] / 2)
        ls_a.append(v)
        ls_b.append(u)
        ls_inter.append(val['weight'] / 2)
    ls_a = np.array(ls_a, dtype=np.int64)
    ls_b = np.array(ls_b, dtype=np.int64)
    ls_inter = np.array(ls_inter, dtype=float)
    W = csr_matrix((ls_inter, (ls_a, ls_b)), shape=(args.n, args.n), dtype=float)
    return W


def sample_discrete_matrix_choices(matrix):
    k, N = matrix.shape
    samples = []

    for j in range(N):
        probabilities = matrix[:, j]
        sample = random.choices(range(k), weights=probabilities, k=1)[0]
        samples.append(sample)

    return np.array(samples)


def mirror_gd(args, obj, f_and_grad, X0, W, L):
    i = 0
    epsilon = args.epsilon_PMD
    np.set_printoptions(suppress=True)
    ls = []
    while True:
        [f, g] = f_and_grad(X0)
        tmp = -1 / (L) * g
        tmp = tmp.astype('float')
        tmp = tmp - tmp.mean(axis=0)
        X = X0 * np.exp(tmp)
        X_norm = X.sum(axis=0)
        denominator_non_zero = (X_norm != 0)
        X = np.where(denominator_non_zero, X / X_norm, 0)
        diff = bregman_kl(X, X0)
        ls.append(diff)
        if diff < epsilon:
            break
        prev = X0
        X0 = X
        i = i + 1
    return X0


def sgd(args, X0, W, L):
    i = 0
    t0 = time.time()
    while True:
        X = project_matrix_to_simplex(X0 - 1 / (L + i + 1) * 2 * X0 @ W)
        diff = bregman_euclidean(X, X0)
        if i % 100 == 0:
            logging.info("diff = " + str(diff))
        if diff < args.epsilon_PMD:
            break
        X0 = X
        i = i + 1
        if time.time() - t0 > 1800:
            return np.inf
    return X0


def project_to_simplex(v, z=1):
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def project_matrix_to_simplex(X):
    result = np.apply_along_axis(project_to_simplex, 0, X)
    return result

def bregman_euclidean(X, Y):
    return np.linalg.norm(X - Y, ord='fro') ** 2 / 2
