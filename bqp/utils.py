import numpy


def get_matrix(args, nx_G):
    n_nodes = args.n
    W_mat = numpy.zeros((n_nodes, n_nodes))
    for (u, v, val) in nx_G.edges(data = True):
        W_mat[u][v] = val['weight'] / 2
        W_mat[v][u] = val['weight'] / 2
    return W_mat