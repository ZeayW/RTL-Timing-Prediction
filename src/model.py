"""Torch Module for TimeConv layer"""

import torch as th
from torch import nn
from dgl import function as fn


class MLP(th.nn.Module):
    def __init__(self, *sizes, negative_slope=0.1, batchnorm=False, dropout=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(th.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(th.nn.LeakyReLU(negative_slope=negative_slope))
                if dropout: fcs.append(th.nn.Dropout(p=0.01))
                if batchnorm: fcs.append(th.nn.BatchNorm1d(sizes[i]))
        self.layers = th.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)


class TimeConv(nn.Module):

    def __init__(self,
                 infeat_dim,
                 hidden_dim):
        super(TimeConv, self).__init__()

        self.hidden_dim = hidden_dim
        self.mlp_pi = MLP(1, int(hidden_dim / 2), hidden_dim)
        self.mlp_agg = MLP(hidden_dim + infeat_dim, int(hidden_dim/2), hidden_dim)
        self.mlp_neigh = MLP(hidden_dim, int(hidden_dim / 2), hidden_dim)
        self.mlp_self= MLP(infeat_dim, int(hidden_dim / 2), hidden_dim)
        self.mlp_global = MLP(1, int(hidden_dim / 2), hidden_dim)
        self.mlp_out = MLP(hidden_dim*2,hidden_dim,1)
        self.activation = nn.ReLU()
        # initialize the parameters
        # self.reset_parameters()


    def nodes_func(self,nodes):
        h = self.mlp_neigh(nodes.data['neigh']) + self.mlp_self(nodes.data['feat'])
        # apply activation except the POs
        mask = nodes.data['is_po'].squeeze() != 1
        h[mask] = self.activation(h[mask])
        return {'h':h}

    def nodes_func_pi(self,nodes):
        h = self.mlp_pi(nodes.data['delay'])
        return {'h':h}

    def forward(self, graph,graph_info):
        topo = graph_info['topo']
        PO_mask = graph_info['POs']
        PO_feat = graph_info['POs_feat']
        with graph.local_scope():
            #propagate messages in the topological order, from PIs to POs
            for i, nodes in enumerate(topo):
                # for PIs
                if i==0:
                    graph.apply_nodes(self.nodes_func_pi,nodes)
                # for other nodes
                else:
                    graph.pull(nodes, fn.copy_src('h','m'), fn.mean('m', 'neigh'), self.nodes_func)

            h_gnn = graph.ndata['h'][PO_mask]
            h_global = self.mlp_global(PO_feat)
            h = th.cat([h_gnn,h_global],dim=1)
            rst = self.mlp_out(h)

            return rst

