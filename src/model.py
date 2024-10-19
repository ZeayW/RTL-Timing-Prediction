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
                 infeat_dim1,
                 infeat_dim2,
                 hidden_dim,
                 agg_choice=0,
                 attn_choice=1,
                 flag_splitfeat=False,
                 flag_homo=False,
                 flag_global=True,
                 flag_attn=False):
        super(TimeConv, self).__init__()

        self.flag_global = flag_global
        self.flag_attn = flag_attn
        self.hidden_dim = hidden_dim
        self.agg_choice = agg_choice
        self.attn_choice = attn_choice
        self.mlp_pi = MLP(1, int(hidden_dim / 2), hidden_dim)
        self.mlp_agg = MLP(hidden_dim, int(hidden_dim / 2), hidden_dim)

        if flag_splitfeat:
            self.feat_name1 = 'feat_gate'
            self.feat_name2 = 'feat_module'

            self.mlp_self_gate = MLP(infeat_dim1, int(hidden_dim / 2), hidden_dim)
            self.mlp_self_module = MLP(infeat_dim2, int(hidden_dim / 2), hidden_dim)
            self.infeat_dim2 = infeat_dim2
            self.infeat_dim1 = infeat_dim1
        else:
            self.feat_name1 = 'feat'
            self.feat_name2 = 'feat'
            self.mlp_self = MLP(infeat_dim1+infeat_dim2, int(hidden_dim / 2), hidden_dim)
            self.mlp_self_module = self.mlp_self
            self.mlp_self_gate = self.mlp_self
            self.infeat_dim2 = infeat_dim2 + infeat_dim1
            self.infeat_dim1 = infeat_dim2 + infeat_dim1
        if flag_homo:
            self.mlp_neigh = MLP(hidden_dim, int(hidden_dim / 2), hidden_dim)
        else:
            if self.agg_choice==0:
                neigh_dim = hidden_dim
            elif self.agg_choice==1:
                neigh_dim = hidden_dim + 1
            elif self.agg_choice == 2:
                neigh_dim = hidden_dim + self.infeat_dim2
            elif self.agg_choice == 3:
                neigh_dim = hidden_dim + self.infeat_dim2 + 1
            self.mlp_neigh_module = MLP(neigh_dim, int(hidden_dim / 2), hidden_dim)
            self.mlp_neigh_gate = MLP(hidden_dim, int(hidden_dim / 2), hidden_dim)
        if flag_global:
            self.mlp_global = MLP(1, int(hidden_dim / 2), hidden_dim)
        if flag_attn:
            atnn_dim_m = hidden_dim
            if self.attn_choice in [1,3]:
                self.mlp_type = MLP(self.infeat_dim2, 32, 32)
                self.mlp_pos = MLP(1, 32, 32)
                atnn_dim_m += 64
            elif self.attn_choice in [2,4]:
                self.mlp_key = MLP(self.infeat_dim2+1, 64, 64)
                atnn_dim_m += 64
            self.attention_vector_g = nn.Parameter(th.randn(hidden_dim, 1), requires_grad=True)
            self.attention_vector_m = nn.Parameter(th.randn(atnn_dim_m,1),requires_grad=True)

        out_dim = hidden_dim*2 if flag_global else hidden_dim
        self.mlp_out = MLP(out_dim,hidden_dim,1)
        self.activation = nn.ReLU()

        # initialize the parameters
        # self.reset_parameters()


    def nodes_func(self,nodes):
        if self.flag_attn:
            #h = self.mlp_neigh(nodes.data['neigh']) + self.mlp_self(nodes.data['feat'])
            h = self.mlp_neigh(nodes.data['neigh'])
        else:
            h = self.mlp_neigh(nodes.data['neigh']) + self.mlp_self(nodes.data[self.feat_name1])
        # apply activation except the POs
        mask = nodes.data['is_po'].squeeze() != 1
        h[mask] = self.activation(h[mask])
        return {'h':h}

    def nodes_func_module(self,nodes):
        # if self.flag_attn:
        #     #h = self.mlp_neigh(nodes.data['neigh']) + self.mlp_self(nodes.data['feat'])
        #     h = self.mlp_neigh_module(nodes.data['neigh'])
        # else:
        mask = nodes.data['is_po'].squeeze() != 1
        if self.agg_choice in [0,1]:
            h = self.mlp_neigh_module(nodes.data['neigh']) + self.mlp_self_module(nodes.data[self.feat_name2])
            h[mask] = self.activation(h[mask])
        elif self.agg_choice in [2,3]:
            h = th.cat((nodes.data['neigh'],nodes.data[self.feat_name2]),dim=1)
            h = self.mlp_neigh_module(h)
            #h[mask] = self.activation(h[mask])
        # apply activation except the POs

        return {'h':h}

    def nodes_func_gate(self,nodes):
        # if self.flag_attn:
        #     #h = self.mlp_neigh(nodes.data['neigh']) + self.mlp_self(nodes.data['feat'])
        #     h = self.mlp_neigh_gate(nodes.data['neigh'])
        # else:
        h = self.mlp_neigh_gate(nodes.data['neigh']) + self.mlp_self_gate(nodes.data[self.feat_name1])
        # apply activation except the POs
        mask = nodes.data['is_po'].squeeze() != 1
        h[mask] = self.activation(h[mask])
        return {'h':h}


    def message_func_attn_module(self,edges):

        #z = self.mlp_attn(th.cat((edges.src['h'],edges.dst['feat']),dim=1))
        #z = th.cat((edges.src['h'],edges.dst['feat']),dim=1)
        #z = edges.src['h']
        if self.attn_choice==0:
            z = edges.src['h']
        elif self.attn_choice in [1,3]:
            z = th.cat((self.mlp_type(edges.dst[self.feat_name2]),self.mlp_pos(edges.data['bit_position'].unsqueeze(1)), edges.src['h']), dim=1)
        elif self.attn_choice in [2,4]:
            z = th.cat((self.mlp_key(th.cat((edges.dst[self.feat_name2],edges.data['bit_position'].unsqueeze(1)),dim=1)), edges.src['h']), dim=1)
        #z = th.cat((edges.data['bit_position'].unsqueeze(1),edges.src['h']),dim=1)
        #z = edges.src['h']
        #z = self.mlp_key(edges.data['bit_position'].unsqueeze(1))
        e = th.matmul(z,self.attention_vector_m)

        if self.agg_choice in [0,2]:
            m = edges.src['h']
        elif self.agg_choice in [1,3]:
            m = th.cat((edges.src['h'],edges.data['bit_position'].unsqueeze(1)),dim=1)

        return {'m':m,'attn_e':e}


    def message_func_attn_gate(self,edges):

        #z = self.mlp_attn(th.cat((edges.src['h'],edges.dst['feat']),dim=1))
        #z = th.cat((edges.src['h'],edges.dst['feat']),dim=1)
        #z = edges.src['h']

        z = edges.src['h']
        e = th.matmul(z,self.attention_vector_g)

        return {'m':edges.src['h'],'attn_e':e}

    def reduce_func_attn(self,nodes):
        alpha = th.softmax(nodes.mailbox['attn_e'],dim=1)
        h = th.sum(alpha*nodes.mailbox['m'],dim=1)

        return {'neigh':h}

    def reduce_func_smoothmax(self, nodes):
        msg = nodes.mailbox['m']
        weight = th.softmax(msg, dim=1)
        return {'neigh': (msg * weight).sum(1)}


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
                isModule_mask = graph.ndata['is_module'][nodes] == 1
                isGate_mask = graph.ndata['is_module'][nodes] == 0

                # for PIs
                if i==0:
                    graph.apply_nodes(self.nodes_func_pi,nodes)
                # for other nodes
                elif self.flag_attn:
                    #graph.pull(nodes, self.message_func_attn, self.reduce_func_attn, self.nodes_func)
                    if graph_info['is_heter']:
                        nodes_gate = nodes[isGate_mask]
                        nodes_module = nodes[isModule_mask]
                        message_func_gate = self.message_func_attn_gate if self.attn_choice in [3,4] else fn.copy_src('h', 'm')
                        reduce_func_gate = self.reduce_func_attn if self.attn_choice in [3,4] else fn.mean('m', 'neigh')

                        if len(nodes_gate)!=0: graph.pull(nodes_gate, message_func_gate, reduce_func_gate, self.nodes_func_gate, etype='intra_gate')
                        if len(nodes_module)!=0: graph.pull(nodes_module, self.message_func_attn_module, self.reduce_func_attn, self.nodes_func_module, etype='intra_module')
                    else:
                        graph.pull(nodes, self.message_func_attn, self.reduce_func_attn, self.nodes_func)
                else:
                    #reduce_func = fn.max('m', 'neigh')
                    #reduce_func = fn.mean('m', 'neigh')
                    #reduce_func = self.reduce_func_smoothmax
                    if graph_info['is_heter']:
                        nodes_gate = nodes[isGate_mask]
                        nodes_module = nodes[isModule_mask]

                        if len(nodes_gate)!=0: graph.pull(nodes_gate, fn.copy_src('h', 'm'), fn.mean('m', 'neigh'), self.nodes_func_gate, etype='intra_gate')
                        if len(nodes_module)!=0: graph.pull(nodes_module, fn.copy_src('h', 'm'), fn.mean('m', 'neigh'), self.nodes_func_module, etype='intra_module')
                    else:
                        graph.pull(nodes, fn.copy_src('h', 'm'), fn.mean('m', 'neigh'), self.nodes_func)


            h_gnn = graph.ndata['h'][PO_mask]

            if self.flag_global:
                h_global = self.mlp_global(PO_feat)

                h = th.cat([h_gnn,h_global],dim=1)
            else:
                h = h_gnn

            rst = self.mlp_out(h)

            return rst

class GraphProp(nn.Module):

    def __init__(self,featname):
        super(GraphProp, self).__init__()
        self.featname = featname

    def nodes_func_delay(self,nodes):
        h = nodes.data['neigh'] + 1
        return {'delay':h}

    def forward(self, graph,topo):
        with graph.local_scope():
            #propagate messages in the topological order, from PIs to POs
            for i, nodes in enumerate(topo[1:]):
                graph.pull(nodes, fn.copy_src(self.featname, 'm'), fn.max('m', self.featname))


            return graph.ndata[self.featname]

