import torch as th
import dgl
from torch import nn
from dgl import function as fn

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

def is_heter(graph):
    return len(graph._etypes)>1 or len(graph._ntypes)>1

def heter2homo(graph):
    src_module, dst_module = graph.edges(etype='intra_module', form='uv')
    src_gate, dst_gate = graph.edges(etype='intra_gate', form='uv')
    homo_g = dgl.graph((th.cat([src_module, src_gate]), th.cat([dst_module, dst_gate])))

    for key, data in graph.ndata.items():
        homo_g.ndata[key] = graph.ndata[key]

    return homo_g

def gen_topo(graph,flag_reverse=False):

    src_module, dst_module = graph.edges(etype='intra_module', form='uv')
    src_gate, dst_gate = graph.edges(etype='intra_gate', form='uv')
    g = dgl.graph((th.cat([src_module,src_gate]), th.cat([dst_module,dst_gate])))
    topo = dgl.topological_nodes_generator(g,reverse=flag_reverse)

    return topo


def add_reverse_edges(graph):
    if is_heter(graph):
        edges_g = graph.edges(etype='intra_gate')
        edges_m = graph.edges(etype='intra_module')
        reverse_edges = (th.cat((edges_g[1],edges_m[1])), th.cat((edges_g[0],edges_m[0])))

        new_graph = dgl.heterograph(
            {
                ('node','intra_module','node'):edges_m,
                ('node', 'intra_gate', 'node'): edges_g,
                ('node', 'reverse', 'node'): reverse_edges
            }
        )
        for key, value in graph.ndata.items():
            new_graph.ndata[key] = value

        for key, value in graph.edges['intra_gate'].data.items():
            new_graph.edges['intra_gate'].data[key] = value

        for key, value in graph.edges['intra_module'].data.items():
            new_graph.edges['intra_module'].data[key] = value

    else:
        new_graph = dgl.heterograph(
            {
                ('node', 'edge', 'node'): graph.edges(),
                ('node', 'edge_r', 'node'): (graph.edges()[1],graph.edges()[0]),
            }
        )
        for key, value in graph.ndata.items():
            new_graph.ndata[key] = value
        for key, value in graph.edata.items():
            new_graph.edges['edge'].data[key] = value
            new_graph.edges['edge_r'].data[key] = value

    return new_graph


def gen_topo(graph, flag_reverse=False):
    src_module, dst_module = graph.edges(etype='intra_module', form='uv')
    src_gate, dst_gate = graph.edges(etype='intra_gate', form='uv')
    g = dgl.graph((th.cat([src_module, src_gate]), th.cat([dst_module, dst_gate])))
    topo = dgl.topological_nodes_generator(g, reverse=flag_reverse)

    return topo


def heter2homo(graph):
    src_module, dst_module = graph.edges(etype='intra_module', form='uv')
    src_gate, dst_gate = graph.edges(etype='intra_gate', form='uv')
    homo_g = dgl.graph((th.cat([src_module, src_gate]), th.cat([dst_module, dst_gate])))

    for key, data in graph.ndata.items():
        homo_g.ndata[key] = graph.ndata[key]

    return homo_g


def reverse_graph(g):
    edges = g.edges()
    reverse_edges = (edges[1], edges[0])

    rg = dgl.graph(reverse_edges, num_nodes=g.num_nodes())
    for key, value in g.ndata.items():
        # print(key,value)
        rg.ndata[key] = value
    for key, value in g.edata.items():
        # print(key,value)
        rg.edata[key] = value
    return rg


def graph_filter(graph):
    homo_graph = heter2homo(graph)
    homo_graph_r = reverse_graph(homo_graph)
    topo_r = dgl.topological_nodes_generator(homo_graph_r)
    graphProp_model = GraphProp('temp')
    homo_graph_r.ndata['temp'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
    homo_graph_r.ndata['temp'][graph.ndata['is_po'] == 1] = 1
    fitler_mask = graphProp_model(homo_graph_r, topo_r).squeeze(1)
    # print(fitler_mask.shape,th.sum(fitler_mask))
    remove_nodes = th.tensor(range(graph.number_of_nodes()))[fitler_mask == 0]
    remain_nodes = th.tensor(range(graph.number_of_nodes()))[fitler_mask == 1]
    # print('\t filtering: remove {} useless nodes'.format(len(remove_nodes)))

    return remain_nodes, remove_nodes


def find_faninPIs(graph):
    fanin_pis = {}
    homo_graph = heter2homo(graph)
    homo_graph_r = reverse_graph(homo_graph)
    topo_r = dgl.topological_nodes_generator(homo_graph_r)
    graphProp_model = GraphProp('po_onehot')
    nodes_list = th.tensor(range(graph.number_of_nodes()))
    POs = nodes_list[graph.ndata['is_po'] == 1].numpy().tolist()
    homo_graph_r.ndata['po_onehot'] = th.zeros((graph.number_of_nodes(), len(POs)), dtype=th.float)
    for i, po in enumerate(POs):
        homo_graph_r.ndata['po_onehot'][po][i] = 1
    mask = graphProp_model(homo_graph_r, topo_r).squeeze(1)
    for i, po in enumerate(POs):
        pi_mask = th.logical_and((mask[:, [i]] == 1).squeeze(1), graph.ndata['is_pi'] == 1)
        fanin_pis[po] = nodes_list[pi_mask].numpy().tolist()

    return fanin_pis


def filter_list(l, idxs):
    new_l = []
    for i in idxs:
        new_l.append(l[i])

    return new_l


