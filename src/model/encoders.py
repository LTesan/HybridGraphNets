"""model.py"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add


# Multi Layer Perceptron (MLP) class
class MLP(torch.nn.Module):
    def __init__(self, layer_vec):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k+1])
            self.layers.append(layer)
            if k != len(layer_vec) - 2: self.layers.append(nn.SiLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Edge model
class EdgeModelW(torch.nn.Module):
    def __init__(self, args):
        super(EdgeModelW, self).__init__()
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.edge_mlp = MLP([3*self.dim_hidden] + self.n_hidden*[self.dim_hidden] + [self.dim_hidden])

    def forward(self, src, dest, edge_attr):
        out = torch.cat([edge_attr, src, dest], dim=1)
        out = self.edge_mlp(out)
        return out

class EdgeModelM(torch.nn.Module):
    def __init__(self, args):
        super(EdgeModelM, self).__init__()
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.edge_mlp = MLP([3*self.dim_hidden] + self.n_hidden*[self.dim_hidden] + [self.dim_hidden])

    def forward(self, src, dest, edge_attr):
        out = torch.cat([edge_attr, src, dest], dim=1)
        out = self.edge_mlp(out)
        return out


# Node model
class NodeModel(torch.nn.Module):
    def __init__(self, args):
        super(NodeModel, self).__init__()
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.node_mlp = MLP([3*self.dim_hidden] + self.n_hidden*[self.dim_hidden] + [self.dim_hidden])

    def forward(self, x, edge_indexw, edge_indexm, edge_attrw, edge_attrm):
        src, dest = edge_indexw
        outw = scatter_add(edge_attrw, dest, dim=0, dim_size=x.size(0))
        src, dest = edge_indexm
        outm = scatter_add(edge_attrm, dest, dim=0, dim_size=x.size(0))
        out = torch.cat([x, outw, outm], dim=1)
        out = self.node_mlp(out)
        return out


# Modification of the original MetaLayer class
class MetaLayer(torch.nn.Module):
    def __init__(self, edge_modelw=None, edge_modelm=None, node_model=None):
        super().__init__()
        self.edge_modelw = edge_modelw
        self.edge_modelm = edge_modelm
        self.node_model = node_model

    def forward(self, x_m, x_w, edge_w, edge_m, edge_attrw, edge_attrm):

        srcw = edge_w[0]
        destw = edge_w[1]

        srcm = edge_m[0]
        destm = edge_m[1]

        edge_attrw = self.edge_modelw(x_w[srcw], x_w[destw], edge_attrw)
        edge_attrm = self.edge_modelm(x_m[srcm], x_m[destm], edge_attrm)
        x = self.node_model(x_m, edge_w, edge_m, edge_attrw, edge_attrm)

        return x, edge_attrw, edge_attrm