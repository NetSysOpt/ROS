import torch
from collections import OrderedDict, defaultdict
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
from itertools import chain
import torch.nn as nn
import networkx as nx
import torch
import time
from .models import GCN_dev
import numpy as np


def gen_q_matrix(args, nx_G):
    Q = torch.zeros((args.n, args.n)).type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)
    sum = np.zeros(args.n)
    for (u, v, w) in nx_G.edges(data=True):
        Q[u][v] = w['weight'] * 2
        sum[u] = sum[u] + w['weight']
        sum[v] = sum[v] + w['weight']
    for u in range(args.n):
        Q[u][u] = -sum[u]
    return Q


def get_gnn(args, gnn_hypers, opt_params, torch_device, torch_dtype, graph):
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    number_classes = gnn_hypers['number_classes']
    
    
    # inputs = torch.rand((args.n, dim_embedding)).type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)
    
    inputs = nn.Embedding(args.n, dim_embedding).type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)
    
    net = GCN_dev(dim_embedding, hidden_dim, number_classes, torch_device)
    net = net.type(torch_dtype).to(torch_device)
    optimizer = torch.optim.Adam(net.parameters(), **opt_params)

    edges = []
    edges_weight = []
    for (u, v, w) in graph.edges(data=True):
        val = w['weight']
        edges.append([u, v])
        edges.append([v, u])
        edges_weight.append(val)
        edges_weight.append(val)
    edges = torch.tensor(edges).transpose(1, 0).to(torch_device)
    edges_weight = torch.tensor(edges_weight).type(torch_dtype).to(torch_device)
    return net, optimizer, edges, edges_weight, inputs


def loss_func(probs, Q_mat):
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined Q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        Q_mat: QUBO as torch tensor
    """

    cost = probs.T @ Q_mat @ probs
    return cost


def run_gnn_training(q_torch, inputs, graph, edges, edges_weight, net, optimizer, number_epochs, tol, patience):
    prev_loss = 1.
    count = 0
    best_loss = np.inf
    t_gnn_start = time.time()

    for epoch in range(number_epochs):

        probs = net(inputs.weight, edges, edges_weight)

        loss = loss_func(probs, q_torch)
        loss_ = loss.detach().item()

        bitstring = (probs.detach() >= 0.5) * 1
        
        if loss_ < best_loss:
            best_loss = loss_
            best_bitstring = bitstring

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_}')
        
        if (abs(loss_ - prev_loss)<=tol) | ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0
        
        if count >= patience:
            print(f"stopping early on epoch {epoch} (patience: {patience})")
            break

        prev_loss = loss_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    t_gnn = time.time() - t_gnn_start
    print(f'GNN training (n={graph.number_of_nodes()}) took {round(t_gnn, 3)}')
    print(f'GNN final continuous loss: {loss_}')
    print(f'GNN best continuous loss: {best_loss}')

    final_bitstring = (probs.detach() >= 0.5) * 1
    best_val = loss_func(best_bitstring.float(), q_torch)
    return net, epoch, final_bitstring, best_bitstring, best_val.item()