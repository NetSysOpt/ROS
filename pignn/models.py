import torch
from torch_geometric.nn import GraphConv
import torch.nn as nn
import torch
from torch_geometric.nn.norm import GraphNorm


class GCN_dev(nn.Module):
    def __init__(self, in_feats, hidden_size, number_classes, device):
        super(GCN_dev, self).__init__()
        
        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.bn1 = GraphNorm(hidden_size)
        self.conv2 = GraphConv(hidden_size, number_classes).to(device)
        self.bn2 = GraphNorm(number_classes)

    def forward(self, inputs, edges, edges_weight):
        h = self.conv1(inputs, edges, edges_weight)
        h = self.bn1(h)
        h = torch.relu(h)
        h = self.conv2(h, edges, edges_weight)
        h = self.bn2(h)
        h = torch.sigmoid(h)
        return h