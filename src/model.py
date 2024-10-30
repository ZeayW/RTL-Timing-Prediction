"""Torch Module for TimeConv layer"""
import dgl
import torch as th
from torch import nn
from dgl import function as fn
from utils import *
from train import device

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
                 pi_choice=0,
                 agg_choice=0,
                 attn_choice=1,
                 flag_reverse=False,
                 flag_splitfeat=False,
                 flag_homo=False,
                 flag_global=True,
                 flag_attn=False):
        super(TimeConv, self).__init__()

        self.pi_choice = pi_choice
        self.flag_reverse = flag_reverse
        self.flag_global = flag_global
        self.flag_attn = flag_attn
        self.hidden_dim = hidden_dim
        self.agg_choice = agg_choice
        self.attn_choice = attn_choice
        self.mlp_pi = MLP(1, int(hidden_dim / 2), hidden_dim)
        self.mlp_agg = MLP(hidden_dim, int(hidden_dim / 2), hidden_dim)

        out_dim = hidden_dim
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
            neigh_dim_m = hidden_dim + self.infeat_dim2 + 1
            neigh_dim_g = hidden_dim + self.infeat_dim1
            self.mlp_neigh_module = MLP(neigh_dim_m, int(hidden_dim / 2), hidden_dim)
            self.mlp_neigh_gate = MLP(neigh_dim_g, int(hidden_dim / 2), hidden_dim)
        if flag_global:
            self.mlp_global = MLP(1, int(hidden_dim / 2), hidden_dim)
            out_dim = hidden_dim * 2
        if flag_attn:
            atnn_dim_m = hidden_dim + 64
            self.mlp_type = MLP(self.infeat_dim2, 32, 32)
            self.mlp_pos = MLP(1, 32, 32)
            self.attention_vector_g = nn.Parameter(th.randn(hidden_dim, 1), requires_grad=True)
            self.attention_vector_m = nn.Parameter(th.randn(atnn_dim_m,1),requires_grad=True)
        # if flag_reverse:
        #     if self.pi_choice==0: self.mlp_global_pi = MLP(1, int(hidden_dim / 2), hidden_dim)
        #     out_dim = hidden_dim * 2


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

        mask = nodes.data['is_po'].squeeze() != 1
        # if self.agg_choice in [0,1]:
        #     h = self.mlp_neigh_module(nodes.data['neigh']) + self.mlp_self_module(nodes.data[self.feat_name2])
        #     h[mask] = self.activation(h[mask])
        # elif self.agg_choice in [2,3]:
        h = th.cat((nodes.data['neigh'],nodes.data[self.feat_name2]),dim=1)
        h = self.mlp_neigh_module(h)
        h[mask] = self.activation(h[mask])

        return {'h':h,'attn_sum':nodes.data['attn_sum']}

    def nodes_func_gate(self,nodes):

        mask = nodes.data['is_po'].squeeze() != 1

        # if self.agg_choice in [0,1]:
        #     h = self.mlp_neigh_gate(nodes.data['neigh']) + self.mlp_self_gate(nodes.data[self.feat_name2])
        # elif self.agg_choice in [2,3]:
        h = th.cat((nodes.data['neigh'],nodes.data[self.feat_name1]),dim=1)
        h = self.mlp_neigh_gate(h)
        h[mask] = self.activation(h[mask])

        return {'h':h,'exp_src_sum':nodes.data['exp_src_sum']}

    def edge_msg_module(self,edges):
        z = th.cat((self.mlp_type(edges.dst[self.feat_name2]), self.mlp_pos(edges.data['bit_position'].unsqueeze(1)),
                    edges.src['h']), dim=1)
        e = th.matmul(z, self.attention_vector_m)

        return {'attn_e':e}

    def edge_msg_module_weight(self,edges):
        normalized_attn_e = th.exp(edges.data['attn_e']) / edges.dst['attn_sum'].squeeze(2)

        return {'weight': normalized_attn_e}

    def message_func_module(self,edges):
        m = th.cat((edges.src['h'], edges.data['bit_position'].unsqueeze(1)), dim=1)

        return {'m':m,'attn_e':edges.data['attn_e']}

    def reduce_func_attn(self,nodes):
        attn_sum = th.sum(th.exp(nodes.mailbox['attn_e']),dim=1).unsqueeze(1)
        # alpha = nodes.mailbox['attn_e'] / attn_sum

        alpha = th.softmax(nodes.mailbox['attn_e'], dim=1)
        h = th.sum(alpha*nodes.mailbox['m'],dim=1)

        return {'neigh':h,'attn_sum':attn_sum}


    def edge_msg_gate_weight(self,edges):

        weight = th.mean(th.exp(edges.src['h']) / edges.dst['exp_src_sum'],dim=1)

        return {'weight': weight}

    def message_func_gate(self,edges):

        z = edges.src['h']
        #z = th.cat((edges.dst['feat'],edges.src['h']), dim=1)
        e = th.matmul(z,self.attention_vector_g)

        return {'m':edges.src['h'],'attn_e':e}

    def reduce_func_smoothmax(self, nodes):
        msg = nodes.mailbox['m']
        weight = th.softmax(msg, dim=1)
        #criticality = th.mean(weight,dim=2)

        return {'neigh': (msg * weight).sum(1),'exp_src_sum':th.sum(th.exp(msg),dim=1)}


    def message_func_reverse(self,edges):

        prob = edges.src['hp'] * edges.data['weight']

        return {'mp':prob}


    def nodes_func_pi(self,nodes):
        h = self.mlp_pi(nodes.data['delay'])
        return {'h':h}

    def prop_backward(self,graph):
        graph.edges['reverse'].data['weight'] = th.cat((graph.edges['intra_gate'].data['weight'].unsqueeze(1),graph.edges['intra_module'].data['weight']))

        num_pos = th.sum(graph.ndata['is_po'])
        graph.ndata['hp'] = th.zeros((graph.number_of_nodes(),num_pos), dtype=th.float).to(device)
        POs = th.tensor(range(graph.number_of_nodes())).to(device)[graph.ndata['is_po']==1].detach().cpu().numpy().tolist()
        for i,po in enumerate(POs):
            graph.ndata['hp'][po][i] = 1
        topo_r = gen_topo(graph,flag_reverse=True)
        topo_r = [l.to(device) for l in topo_r]

        # new_PIs = []
        with graph.local_scope():
            # POs = topo_r[0]
            # for po in POs.cpu().numpy().tolist():
            #     if len(graph.in_edges(po,form='eid',etype='intra_gate')) + len(graph.in_edges(po,form='eid',etype='intra_module'))==0:
            #         new_PIs.append(po)
            for i, nodes in enumerate(topo_r[1:]):
                graph.pull(nodes, self.message_func_reverse, fn.sum('mp', 'hp'), etype='reverse')
                # for n in nodes.cpu().numpy().tolist():
                #     if len(graph.in_edges(n,form='eid',etype='intra_gate')) + len(graph.in_edges(n,form='eid',etype='intra_module'))==0:
                #         new_PIs.append(n)
            return graph.ndata['hp']

    def forward(self, graph,graph_info):
        topo = graph_info['topo']
        PO_mask = graph_info['POs']
        PO_feat = graph_info['POs_feat']
        #if True:
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
                    if graph_info['is_heter']:
                        nodes_gate = nodes[isGate_mask]
                        nodes_module = nodes[isModule_mask]
                        message_func_gate = self.message_func_attn_gate if self.attn_choice==0 else fn.copy_src('h', 'm')
                        reduce_func_gate = self.reduce_func_attn if self.attn_choice==0 else self.reduce_func_smoothmax

                        if len(nodes_gate)!=0:
                            graph.pull(nodes_gate, message_func_gate, reduce_func_gate, self.nodes_func_gate, etype='intra_gate')
                            if self.flag_reverse:
                                eids = graph.in_edges(nodes_gate, form='eid', etype='intra_gate')
                                graph.apply_edges(self.edge_msg_gate_weight, eids, etype='intra_gate')

                        if len(nodes_module)!=0:
                            eids = graph.in_edges(nodes_module, form='eid', etype='intra_module')
                            graph.apply_edges(self.edge_msg_module, eids, etype='intra_module')
                            graph.pull(nodes_module, self.message_func_module, self.reduce_func_attn, self.nodes_func_module, etype='intra_module')
                            if self.flag_reverse:
                                graph.apply_edges(self.edge_msg_module_weight, eids, etype='intra_module')

                    else:
                        graph.pull(nodes, self.message_func_attn, self.reduce_func_attn, self.nodes_func)
                else:
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

            h_pi = None
            if self.flag_reverse:
                nodes_prob = self.prop_backward(graph)
                PIs_mask = graph.ndata['is_pi'] == 1
                PIs_prob = nodes_prob[PIs_mask]
                #print(th.sum(graph.ndata['is_po']),th.sum(nodes_prob[PIs_mask],dim=0))
                if self.pi_choice==0:
                    PIs_delay = graph.ndata['delay'][PIs_mask]
                    pi2po_delay = th.matmul(th.transpose(PIs_prob, 0, 1), PIs_delay)
                    h_pi = self.mlp_global_pi(pi2po_delay)
                elif self.pi_choice==1:
                    PIs_embedding = graph.ndata['h'][PIs_mask]
                    h_pi = th.matmul(th.transpose(PIs_prob, 0, 1), PIs_embedding)
                else:
                    assert False
                #print(th.sum(h_pi))
                h = th.cat([h, h_pi], dim=1)
            rst = self.mlp_out(h)
            #print('g',num_gate,th.sum(graph.edges['intra_gate'].data['weight']))
            #print('m',num_module,th.sum(graph.edges['intra_module'].data['weight']))

            return rst

class GraphBackProp(nn.Module):

    def __init__(self,featname):
        super(GraphBackProp, self).__init__()
        self.featname = featname

    def nodes_func_delay(self,nodes):
        h = nodes.data['neigh'] + 1
        return {'delay':h}

    def forward(self, graph):
        topo_r = dgl.topological_nodes_generator(graph,reverse=True)
        with graph.local_scope():
            #propagate messages in the topological order, from PIs to POs
            for i, nodes in enumerate(topo_r[1:]):
                isModule_mask = graph.ndata['is_module'][nodes] == 1
                isGate_mask = graph.ndata['is_module'][nodes] == 0
                nodes_gate = nodes[isGate_mask]
                nodes_module = nodes[isModule_mask]



                graph.pull(nodes, fn.copy_src(self.featname, 'm'), fn.max('m', self.featname))



            return graph.ndata[self.featname]




