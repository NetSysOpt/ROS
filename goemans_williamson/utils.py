import torch



def get_matrix(args, nx_G):
    n_nodes = len(nx_G.nodes)
    W_mat = torch.zeros(n_nodes, n_nodes)
    for (u, v, val) in nx_G.edges(data = True):
        W_mat[u][v] = val['weight'] / 2
        W_mat[v][u] = val['weight'] / 2
        
    W_mat = W_mat.type(args.TORCH_DTYPE)
    return W_mat