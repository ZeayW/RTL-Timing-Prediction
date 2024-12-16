"""Torch Module for TimeConv layer"""
import dgl
import torch as th
from torch import nn
from dgl import function as fn
from utils import *
from train import device

def get_nodename(nodes_name,nid):
    if nodes_name[nid][1] is not None:
        return nodes_name[nid][1]
    else:
        return nodes_name[nid][0]

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
                 flag_train=True,
                 flag_path_supervise=False,
                 flag_filter=False,
                 flag_reverse=False,
                 flag_splitfeat=False,
                 flag_homo=False,
                 flag_global=True,
                 flag_attn=False):
        super(TimeConv, self).__init__()

        self.pi_choice = pi_choice
        self.flag_train = flag_train
        self.flag_path_supervise = flag_path_supervise
        self.flag_filter = flag_filter
        self.flag_reverse = flag_reverse
        self.flag_global = flag_global
        self.flag_attn = flag_attn
        self.hidden_dim = hidden_dim
        self.agg_choice = agg_choice
        self.attn_choice = attn_choice
        self.mlp_pi = MLP(4, int(hidden_dim / 2), hidden_dim)
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
            hidden_dim_attn = int(hidden_dim/8)
            atnn_dim_m = hidden_dim + hidden_dim_attn *2
            self.mlp_type = MLP(self.infeat_dim2, hidden_dim_attn, hidden_dim_attn)
            self.mlp_pos = MLP(1, hidden_dim_attn, hidden_dim_attn)
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

        if self.flag_reverse:
            return {'h':h,'attn_sum':nodes.data['attn_sum']}
        else:
            return {'h': h}

    def nodes_func_gate(self,nodes):

        mask = nodes.data['is_po'].squeeze() != 1

        # if self.agg_choice in [0,1]:
        #     h = self.mlp_neigh_gate(nodes.data['neigh']) + self.mlp_self_gate(nodes.data[self.feat_name2])
        # elif self.agg_choice in [2,3]:
        h = th.cat((nodes.data['neigh'],nodes.data[self.feat_name1]),dim=1)
        h = self.mlp_neigh_gate(h)
        h[mask] = self.activation(h[mask])
        if self.flag_reverse:
            return {'h': h, 'exp_src_sum': nodes.data['exp_src_sum'],'exp_src_max':nodes.data['exp_src_max']}
        else:
            return {'h':h}

    def edge_msg_module(self,edges):
        z = th.cat((self.mlp_type(edges.dst[self.feat_name2]), self.mlp_pos(edges.data['bit_position'].unsqueeze(1)),
                    edges.src['h']), dim=1)
        e = th.matmul(z, self.attention_vector_m)
        e = e - th.max(e, dim=1).values.unsqueeze(1)
        #print(th.max(e))
        #e = e- th.max(e)

        return {'attn_e':e}

    def edge_msg_module_weight(self,edges):
        normalized_attn_e = th.exp(edges.data['attn_e']) / edges.dst['attn_sum'].squeeze(2)

        if th.isnan(normalized_attn_e).any() or th.isinf(normalized_attn_e).any():
            print('module weight')
            print('isnan:{}, isinf:{}'.format(th.isnan(normalized_attn_e).any(),th.isinf(normalized_attn_e).any()))
            #print('\t attn_e',normalized_attn_e.detach().cpu().numpy().tolist())
            print('\t isinf fenzi',th.isinf(th.exp(edges.data['attn_e'])).any())
            print('\t isinf fenmu', th.isinf(edges.dst['attn_sum']).any())
            print('\t #0 in fenmu',len(edges.dst['attn_sum'][edges.dst['attn_sum']==0]))
            print('\t #0 in fenzi', len(th.exp(edges.data['attn_e'])[th.exp(edges.data['attn_e']) == 0]))
            #print(th.exp(edges.data['attn_e']).detach().cpu().numpy().tolist())

            exit()
        return {'weight': normalized_attn_e}

    def message_func_module(self,edges):
        m = th.cat((edges.src['h'], edges.data['bit_position'].unsqueeze(1)), dim=1)
        #return {'m': m, 'attn_e': edges.data['attn_e']}

        # z = th.cat((self.mlp_type(edges.dst[self.feat_name2]), self.mlp_pos(edges.data['bit_position'].unsqueeze(1)),
        #             edges.src['h']), dim=1)
        # e = th.matmul(z, self.attention_vector_m)
        #e = e - th.max(e, dim=1).values.unsqueeze(1)
        return {'m':m,'attn_e':edges.data['attn_e']}


    def reduce_func_attn(self,nodes):
        #attn_e = nodes.mailbox['attn_e'] - th.max(nodes.mailbox['attn_e'],dim=1)
        # print(th.max(nodes.mailbox['attn_e'],dim=1).value().shape)
        attn_sum = th.sum(th.exp(nodes.mailbox['attn_e']),dim=1).unsqueeze(1)
        # print(len(nodes))
        # print(nodes.mailbox['attn_e'].shape)
        # print(attn_sum.shape)
        # exit()
        # alpha = nodes.mailbox['attn_e'] / attn_sum
        alpha = th.softmax(nodes.mailbox['attn_e'], dim=1)
        h = th.sum(alpha*nodes.mailbox['m'],dim=1)

        if self.flag_reverse:
            attn_sum = th.sum(th.exp(nodes.mailbox['attn_e']), dim=1).unsqueeze(1)
            return {'neigh':h,'attn_sum':attn_sum}
        else:
            return {'neigh': h}


    def edge_msg_gate_weight(self,edges):

        src_msg = edges.src['h'] - th.max(edges.src['h'])
        src_msg = edges.src['h'] - th.max(edges.src['h'],dim=1).values.unsqueeze(1)
        #src_msg = edges.src['h'] - 1000
        src_msg = edges.src['h'] - edges.dst['exp_src_max']
        weight = th.mean(th.exp(src_msg) / edges.dst['exp_src_sum'],dim=1)
        # print(src_msg.shape)
        # print(th.max(edges.src['h'],dim=1).values.unsqueeze(1).shape)
        # exit()
        if th.isnan(weight).any():
            print('gate weight')
            print('isnan:{}, isinf:{}'.format(th.isnan(th.exp(src_msg)).any(),th.isinf(edges.dst['exp_src_sum']).any()))
            print('\t isinf fenzi', th.isinf(th.exp(src_msg)).any())
            print('\t isinf fenmu', th.isinf(edges.dst['exp_src_sum']).any())
            print('\t #0 in fenmu', len(edges.dst['exp_src_sum'][edges.dst['exp_src_sum'] == 0]))
            print('\t #0 in fenzi', len(th.exp(src_msg)[th.exp(src_msg)==0]))
            exit()


        return {'weight': weight}

    def message_func_gate(self,edges):

        z = edges.src['h']
        #z = th.cat((edges.dst['feat'],edges.src['h']), dim=1)
        e = th.matmul(z,self.attention_vector_g)

        return {'m':edges.src['h'],'attn_e':e}

    def reduce_func_smoothmax(self, nodes):
        msg = nodes.mailbox['m']
        #msg = msg - th.max(msg)
        # msg = msg - 1000
        weight = th.softmax(msg, dim=1)
        #criticality = th.mean(weight,dim=2)
        exp_src_max = th.max(msg,dim=1).values
        exp_src_sum = th.sum(th.exp(msg-exp_src_max.unsqueeze(1)),dim=1)

        if self.flag_reverse:
            return {'neigh': (msg * weight).sum(1),'exp_src_sum':exp_src_sum,'exp_src_max':exp_src_max}
        else:
            return {'neigh': (msg * weight).sum(1)}

    def message_func_reverse(self,edges):

        prob = edges.src['hp'] * edges.data['weight']

        if th.isinf(edges.data['weight']).any():
            print('aaa')

        if th.isnan(edges.src['hp']).any() or th.isinf(edges.src['hp']).any():
            print('bbb')

        if th.isnan(th.sum(prob, dim=1)).any():
            print('ccc')
            exit()
        # dst = edges.src['hd'] + 1
        # dst_g = edges.src['hdg']
        # dst_m = edges.src['hdm']
        # dst_m[edges.src['is_module']==1] = dst_m[edges.src['is_module']==1] + 1
        # dst_g[edges.src['is_module'] == 0] = dst_g[edges.src['is_module'] == 0] + 1
        return {'mp':prob}
        #return {'mp': prob, 'dst': edges.src['hd'] + 1}
        #return {'mp':prob,'dst':edges.src['hd']+1,'dst_m':dst_m,'dst_g':dst_g}

    def message_func_loss(self, edges):

        pi_prob = th.gather(edges.src['hp'],dim=1,index=edges.dst['id'])
        # print('hp',edges.src['hp'][:10].detach().cpu().numpy().tolist())
        # print('id',edges.dst['id'][:10].detach().cpu().numpy().tolist())
        # print('prob',pi_prob[:10].detach().cpu().numpy().tolist())
        # print('label',edges.data['prob'][:10])

        return {'ml':pi_prob}
        # return {'ml':th.abs(pi_prob-edges.data['prob'])}

    def reduce_func_reverse(self,nodes):
        #print(th.max(nodes.mailbox['dst'],dim=1).values,len(nodes))

        return {'hp':th.sum(nodes.mailbox['mp'],dim=1),'hd':th.max(nodes.mailbox['dst'],dim=1).values}

    def nodes_func_pi(self,nodes):
        #h = nodes.data['delay']
        h = th.cat((nodes.data['delay'],nodes.data['value']),dim=1)
        h = self.mlp_pi(h)
        mask = nodes.data['is_po'].squeeze() != 1
        #h[mask] = self.activation(h[mask])

        return {'h':h}

    def prop_backward(self,graph,POs):
        graph.edges['reverse'].data['weight'] = th.cat((graph.edges['intra_gate'].data['weight'].unsqueeze(1),graph.edges['intra_module'].data['weight']))
        #print(th.isinf(graph.edges['intra_gate'].data['weight']).any(),th.isinf(graph.edges['intra_module'].data['weight']).any())
        graph.ndata['hp'] = th.zeros((graph.number_of_nodes(),len(POs)), dtype=th.float).to(device)
        #graph.ndata['hd'] = -1000*th.ones((graph.number_of_nodes(), len(POs)), dtype=th.float).to(device)
        for i,po in enumerate(POs):
            # if not self.flag_filter or predicted_labels_l[i]>2:
            graph.ndata['hp'][po][i] = 1
            #graph.ndata['hd'][po][i] = 0
        topo_r = gen_topo(graph,flag_reverse=True)
        topo_r = [l.to(device) for l in topo_r]

        # new_PIs = []
        with graph.local_scope():
            # POs = topo_r[0]
            # for po in POs.cpu().numpy().tolist():
            #     if len(graph.in_edges(po,form='eid',etype='intra_gate')) + len(graph.in_edges(po,form='eid',etype='intra_module'))==0:
            #         new_PIs.append(po)
            for i, nodes in enumerate(topo_r[1:]):
                graph.pull(nodes, self.message_func_reverse, fn.sum('mp','hp'), etype='reverse')
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

            rst = self.mlp_out(h)

            if not self.flag_train and self.flag_path_supervise:
                return rst,None

            if self.flag_reverse:
                POs = th.tensor(range(graph.number_of_nodes())).to(device)[graph.ndata['is_po'] == 1]
                critical_po_mask = rst.squeeze(1) > 10
                # if self.flag_filter:
                #     POs = POs[critical_po_mask]
                POs = POs.detach().cpu().numpy().tolist()
                nodes_prob = self.prop_backward(graph,POs)

                if self.flag_path_supervise:
                    graph.ndata['hp'] = nodes_prob
                    graph.ndata['id'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.int64).to(device)
                    graph.ndata['id'][POs] = th.tensor(range(len(POs)), dtype=th.int64).unsqueeze(-1).to(device)
                    graph.pull(POs, self.message_func_loss, fn.sum('ml', 'loss'), etype='pi2po')
                    # print([get_nodename(graph_info['nodes_name'],po) for po in POs])

                    # PIs_mask = graph.ndata['is_pi'] == 1
                    # PIs = th.tensor(range(graph.number_of_nodes())).to(device)[graph.ndata['is_pi'] == 1]
                    # PIs = PIs.detach().cpu().numpy().tolist()
                    # print('#PI:',len(PIs))
                    # PIs_prob = th.transpose(nodes_prob[PIs_mask], 0, 1)
                    # print(PIs_prob.shape)
                    # for i, po in enumerate(POs):
                    #     pis_prob = PIs_prob[i]
                    # print(get_nodename(graph_info['nodes_name'],po))
                    # print('\t',[(graph_info['nodes_name'][pi][0],prob.item()) for pi,prob in zip(PIs,pis_prob)])
                    # print('\t',th.max(pis_prob))
                    # print('\t',th.sum(pis_prob))
                    # POs_argmaxPI = th.argmax(PIs_prob, dim=1).cpu().numpy().tolist()

                    # print([get_nodename(graph_info['nodes_name'],i) for i in POs_argmaxPI])

                    # exit()
                    path_loss = th.mean(graph.ndata['loss'][POs])

                    if th.isnan(path_loss).any():
                        # print(nodes_prob)
                        print(len(nodes_prob), len(nodes_prob[th.isnan(nodes_prob)]))
                        print(len(graph.ndata['loss'][POs][th.isnan(graph.ndata['loss'][POs])]))
                        # print(graph.ndata['loss'][POs])

                    return rst, path_loss

                # nodes_dst[nodes_dst<-100] = 0


                #PIs_mask = graph.ndata['is_pi'] == 1
                #PIs_mask = th.logical_or(graph.ndata['is_pi'] == 1, graph.ndata['is_module'] == 1)
                #module_mask = graph.ndata['is_module'] == 1
                #modules_prob = nodes_prob[module_mask]
                #POs_moduleprob = th.transpose(modules_prob, 0, 1)
                #h_module = th.matmul(POs_moduleprob, graph.ndata['h'][module_mask])

                PIs_mask = graph.ndata['is_pi'] == 1
                PIs_prob = th.transpose(nodes_prob[PIs_mask], 0, 1)
                POs_argmaxPI = th.argmax(PIs_prob,dim=1)
                #PIs_dst =th.transpose(nodes_dst[PIs_mask], 0, 1)
                critical_PIdelay = graph.ndata['delay'][PIs_mask][POs_argmaxPI]
                weighted_PIdelay = th.matmul(PIs_prob, graph.ndata['delay'][PIs_mask])
                #critical_PIdst = th.gather(PIs_dst,dim=1,index=POs_argmaxPI.unsqueeze(-1))
                #weighted_PIdst = th.sum(PIs_prob*PIs_dst,dim=1).unsqueeze(-1)
                #print(weighted_PIdelay.shape,weighted_PIdst.shape)
                critical_PIinfo = critical_PIdelay
                weighted_PIinfo = weighted_PIdelay
                #critical_PIinfo = th.cat((critical_PIdelay,critical_PIdst),dim=1)
                #weighted_PIinfo = th.cat((weighted_PIdelay,weighted_PIdst),dim=1)

                # critical_PIdelay = critical_PIdelay.squeeze(1).detach().cpu().numpy().tolist()
                # weighted_PIdelay = weighted_PIdelay.squeeze(1).detach().cpu().numpy().tolist()
                # critical_PIdst = critical_PIdst.squeeze(1).detach().cpu().numpy().tolist()
                # weighted_PIdst = weighted_PIdst.squeeze(1).detach().cpu().numpy().tolist()
                # label = graph_info['labels'].squeeze(1).detach().cpu().numpy().tolist()
                # for i in range(len(POs)):
                #     print('{} {:.2f} {} {:.2f} {}'.format(critical_PIdelay[i],weighted_PIdelay[i],critical_PIdst[i],weighted_PIdst[i],label[i]))

                if self.pi_choice==0:
                    # PIs_delay = graph.ndata['delay'][PIs_mask]
                    # pi2po_delay = th.matmul(th.transpose(PIs_prob, 0, 1), PIs_delay)
                    #h_pi = weighted_PIdelay
                    #h_pi = critical_PIdelay
                    h_pi = th.cat((critical_PIinfo,weighted_PIinfo),dim=1)
                    h_pi = self.mlp_global_pi(h_pi)

                elif self.pi_choice==1:
                    h_pi = graph.ndata['h'][PIs_mask][POs_argmaxPI]
                    #h_pi = th.matmul(POs_PIprob, graph.ndata['h'][PIs_mask])

                else:
                    assert False

                #rst = self.mlp_out_new(h)
                #return rst

                if self.flag_filter:
                    # h_critical = th.cat([h[critical_po_mask], h_pi], dim=1)
                    # rst[critical_po_mask] = self.mlp_out_new(h_critical)
                    h_pi[critical_po_mask] = 0
                    rst = self.mlp_out_new(h)
                    #rst[critical_po_mask] = h[critical_po_mask] + self.mlp_out_new(h_pi)
                else:
                    # h = th.cat([h, h_pi], dim=1)
                    # rst = self.mlp_out_new(h)
                    #h_global = th.cat([h_pi, h_module], dim=1)
                    h_global = h_pi
                    rst = rst + self.mlp_out_new(h_global)
                    rst = self.activation(rst)



            #print('g',num_gate,th.sum(graph.edges['intra_gate'].data['weight']))
            #print('m',num_module,th.sum(graph.edges['intra_module'].data['weight']))

            return rst,None

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




