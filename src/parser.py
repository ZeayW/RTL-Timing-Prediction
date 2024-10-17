import torch as th
import dgl
import os
import pickle
from options import get_options
from random import shuffle
from model import GraphProp
import tee


options = get_options()
rawdata_path = options.rawdata_path
data_savepath = options.data_savepath
os.makedirs(data_savepath,exist_ok=True)

ntype_file = os.path.join(data_savepath,'ntype2id.pkl')

ntype2id = {'input':0, "1'b0":1, "1'b1":2}
ntype2id_gate = {'input':0, "1'b0":1, "1'b1":2}
ntype2id_module = {}

class GraphInfo:
    def __init__(self,design_name,case_name):
        self.design_name = design_name
        self.case_name = case_name
        self.src_nodes = []
        self.dst_nodes = []
        self.node2nid = {}
        self.nid2node = {}
        self.nodes_type = []
        self.nodes_delay = []
        self.nodes_name = []
        self.POs = []
        self.POs_label = []
        self.POs_name = []
        self.is_po = []
        self.is_pi = []
        self.POs_feat = None

    def get_info(self):
        return {
            'design_name':self.design_name,
            'case_name':self.case_name,
            "nodes_name":self.nodes_name,
            'POs':self.POs,
            'POs_feat':self.POs_feat,

        }

def gen_topo(graph):

    src_module, dst_module = graph.edges(etype='intra_module', form='uv')
    src_gate, dst_gate = graph.edges(etype='intra_gate', form='uv')
    g = dgl.graph((th.cat([src_module,src_gate]), th.cat([dst_module,dst_gate])))
    topo = dgl.topological_nodes_generator(g)

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
    fitler_mask = graphProp_model(homo_graph_r,topo_r).squeeze(1)
    #print(fitler_mask.shape,th.sum(fitler_mask))
    remove_nodes = th.tensor(range(graph.number_of_nodes()))[fitler_mask==0]
    remain_nodes = th.tensor(range(graph.number_of_nodes()))[fitler_mask==1]
    #print('\t filtering: remove {} useless nodes'.format(len(remove_nodes)))

    return remain_nodes,remove_nodes


def filter_list(l,idxs):
    new_l = []
    for i in idxs:
        new_l.append(l[i])

    return new_l



class Parser:
    def __init__(self,subdir,design_path):
        self.case_name = os.path.split(design_path)[-1]
        self.design_name = '[{}]_{}'.format(subdir,self.case_name.split('_')[-2])
        self.design_path = design_path
        self.wires_width = {}
        self.const0_index = 0
        self.const1_index = 0
        self.nodes = {}
        self.edges = [] # edges: List[Tuple[str, str, Dict]] = []  # a list of (src, dst, {key: value})

        self.pi_delay = {}
        self.po_labels = {}
        pass


    #.i0({3'b000,do_2_b5_n,do_2_b5_n1,do_2_b5_n2,do_2_b5_n3,do_2_b5_n4}),
    def parse_wire(self,wire):

        if wire.strip().startswith('{'):
            res = []
            wires = wire[wire.find('{') + 1:wire.find('}')]
            wires = wires.split(',')
            for w in wires:
                res.extend(self.parse_wire(w.strip()))
        elif "'b" in wire:
            res = []
            values = wire[wire.find("b") + 1:]
            for v in values:
                if v == "0":
                    self.const0_index += 1
                    node = "1'b0_{}".format(self.const0_index)
                    self.nodes[node] = {'ntype': "1'b0", 'is_po': False,'is_module':0}
                    res.append(node)
                elif v == "1":
                    self.const1_index += 1
                    node = "1'b1_{}".format(self.const1_index)
                    self.nodes[node] = {'ntype': "1'b1", 'is_po': False,'is_module':0}
                    res.append(node)
                else:
                    assert False
        elif ':' in wire:

            width = wire[wire.find('[')+1:wire.find(')')]
            high_bit, low_bit = width.split(':')
            high_bit = int(high_bit.strip())
            low_bit = int(low_bit.strip())
            wire_name = wire.split('[')[0]
            res = ["{}[{}]".format(wire_name, b) for b in range(high_bit, low_bit - 1, -1)]
        elif '[' in wire:
            res = [wire]
        elif self.wires_width.get(wire, None) is None:
            res = [wire]
        else:
            low_bit, high_bit = self.wires_width[wire]

            if low_bit == 0 and high_bit == 0:
                res = [wire]
            else:
                res = ["{}[{}]".format(wire, b) for b in range(high_bit, low_bit-1,-1)]


        return res

    def parse_verilog(self):
        file_path = os.path.join(self.design_path,'{}_case.v'.format(self.case_name[:self.case_name.rfind('_')]))

        with open(file_path, 'r') as f:
            content = f.read()
        content = content[content.find('input'):]

        buf_o2i = {}

        for sentence in content.split(';\n'):
            if len(sentence) == 0 or 'endmodule' in sentence:
                continue

            # definition of io/wires
            if 'input ' in sentence or 'output ' in sentence or 'wire ' in sentence:
                # e.g., wire do_2_b14_n;
                if '[' not in sentence:
                    sentence = sentence.strip()
                    wire_type, wire_name = sentence.split(' ')
                    node_name = wire_name.replace(';','')
                    wire_type = wire_type.replace(' ','')
                    self.nodes[node_name] = {'ntype':wire_type,'is_po':wire_type=='output','is_module':0}
                    self.wires_width[node_name] = (0,0)

                # e.g., wire [4:0] do_0_b;
                else:
                    sentence = sentence.strip()
                    wire_type, bit_range, wire_name = sentence.split(' ');
                    high_bit,low_bit = bit_range[bit_range.find('[')+1:bit_range.rfind(']')].split(':')
                    wire_type = wire_type.replace(' ','')
                    wire_name = wire_name.replace(';','')
                    self.wires_width[wire_name] = (int(low_bit),int(high_bit))
                    if int(low_bit)==0 and int(high_bit)==0:
                        self.nodes[wire_name] = {'ntype': wire_type, 'is_po': wire_type == 'output','is_module':0}
                    # else:
                    for i in range(int(low_bit),int(high_bit)+1):
                        node_name = '{}[{}]'.format(wire_name,i)
                        self.nodes[node_name] = {'ntype': wire_type,'is_po':wire_type=='output','is_module':0}




            else:
                fo2fi = {}  # {fanout_name:fanin_list}
                sentence = sentence.replace('\n','').strip()
                # print(sentence)
                # get the gate type
                gate = sentence[:sentence.find('(')]
                gate_type, gate_name = gate.strip().split(' ')

                fanins_bit_position = {}

                # deal with multiplexer, whose width may be larger than 1
                if 'mux' in gate_type.lower():
                    gate_type = 'mux'
                    # get the io wires list
                    io_wires = sentence[sentence.find('(') + 1:].strip()
                    io_wires = io_wires.split('),')
                    io_wires = [p.replace(' ', '') for p in io_wires]

                    io_wires = {p[1:p.find('(')] : p[p.find('(')+1 :].strip().replace(')','') for p in io_wires}

                    io_nodes = {p: self.parse_wire(w) for p,w in io_wires.items()}
                    # print('\t',io_wires)
                    # print('\t',io_nodes)
                    assert len(io_wires) == 4
                    assert io_wires.get('o',None) is not None

                    # get the output nodes, and set their gate type;
                    fanout_nodes = io_nodes['o']
                    for n in fanout_nodes:
                        if self.nodes.get(n,None) is None and 'open' in n:
                            self.nodes[n] = {'ntype':gate_type,'is_po':False,'is_module':0}
                        else:
                            self.nodes[n]['ntype'] = gate_type
                        ntype2id['mux'] = ntype2id.get('mux',len(ntype2id))
                        ntype2id_gate['mux'] = ntype2id_gate.get('mux', len(ntype2id_gate))
                        fo2fi[n] = []
                    # add the edges between fanin nodes and fanout nodes
                    for port, fanin_nodes in io_nodes.items():
                        if port == 'o':
                            continue
                        # for port i0,i1: link fi_i[j] with fo[j]
                        elif 'i' in port:
                            for i,fanout_node in enumerate(fanout_nodes):
                                fo2fi[fanout_node].append(fanin_nodes[i])
                        # for port sel: link all fi_s with fo[j]
                        elif port=='sel':
                            for i, fanout_node in enumerate(fanout_nodes):
                                fo2fi[fanout_node].extend(fanin_nodes)
                        else:
                            assert False
                # deal with arithmetic blocks, whose width may be larger than 1
                elif '.' in sentence:
                    gate_type = gate_type.split('_')[0]
                    io_wires = sentence[sentence.find('(') + 1:].strip()
                    io_wires = io_wires.split('),')
                    io_wires = [p.replace(' ', '') for p in io_wires]
                    io_wires = {p[1:p.find('(')]: p[p.find('(') + 1:].strip().replace(')', '') for p in io_wires}
                    # print(sentence)
                    # print(io_wires)
                    io_nodes = {p: self.parse_wire(w) for p, w in io_wires.items()}
                    assert io_wires.get('o', None) is not None
                    fanout_nodes = io_nodes['o']
                    fanout_nodes.reverse()

                    for n in fanout_nodes:
                        if self.nodes.get(n, None) is None and 'open' in n:
                            self.nodes[n] = {'ntype': gate_type, 'is_po': False,'is_module':1}

                        else:
                            self.nodes[n]['ntype'] = gate_type
                            self.nodes[n]['is_module'] = 1
                        ntype2id[gate_type] = ntype2id.get(gate_type, len(ntype2id))
                        ntype2id_module[gate_type] = ntype2id_module.get(gate_type, len(ntype2id_module))
                        fo2fi[n] = []

                    # add the edges between fanin nodes and fanout nodes
                    for port, fanin_nodes in io_nodes.items():
                        for idx,fi in enumerate(fanin_nodes):
                            fanins_bit_position[fi] = len(fanin_nodes) - idx
                        fanin_nodes.reverse()
                        if port == 'o':
                            continue

                        # for port i0,i1: link fi_i[j...0] with fo[j]
                        elif port.startswith('i') and len(fanout_nodes)!=1 and gate_type not in ['encoder','decoder']:

                            for i, fanout_node in enumerate(fanout_nodes):
                                fo2fi[fanout_node].extend(fanin_nodes[:i+1])
                        # for port sel or one output modules, e.g., eq, lt: link all fi_s with each fo
                        else:
                            for i, fanout_node in enumerate(fanout_nodes):
                                fo2fi[fanout_node].extend(fanin_nodes)
                        # else:
                        #     assert False
                    # for i, fanout_node in enumerate(fanout_nodes):
                    #     print('\t',fanout_node,fo2fi[fanout_node])
                    # print("\n\n")
                    # print(sentence,fo2fi,fanins_bit_position)
                    # exit()

                    pass
                # deal with other one-output gates, e.g., or, and...
                else:
                    # get the paramater list
                    io_wires = sentence[sentence.find('(') + 1:]
                    io_wires = io_wires.replace(')', '').split(',')
                    io_wires = [p.replace(' ','') for p in io_wires]
                    fanout_node = io_wires[0]  # fanout is the first parameter
                    self.nodes[fanout_node]['ntype'] = gate_type
                    fanin_nodes = [self.parse_wire(w)[0] for w in io_wires[1:]]


                    if 'buf' in gate_type:
                        fanin_node = fanin_nodes[0]
                        buf_o2i[fanout_node] = fanin_nodes[0]
                        self.nodes[fanin_node]['nickname'] = fanout_node
                        self.nodes[fanin_node]['is_po'] = self.nodes[fanout_node]['is_po']
                        self.nodes[fanout_node]['ntype'] = None
                        #self.nodes[fanin_node]['is_module'] = self.nodes[fanout_node]['is_module']
                    else:
                        ntype2id[gate_type] = ntype2id.get(gate_type, len(ntype2id))
                        ntype2id_gate[gate_type] = ntype2id_gate.get(gate_type, len(ntype2id_gate))
                        fo2fi[fanout_node] = fanin_nodes

                # add the edges from fanins to fanouts
                for fanout, fanins in fo2fi.items():
                    for fanin in fanins:

                        self.edges.append(
                            (fanin,fanout,{'bit_position':fanins_bit_position.get(fanin,None)})
                        )





        self.nodes = {n : self.nodes[n] for n in self.nodes.keys() if self.nodes[n]['ntype'] not in ['wire',None]}

        is_linked = {}
        new_edges = []
        for src,dst, e_info in self.edges:
            if buf_o2i.get(src,None) is not None:
                new_edges.append((buf_o2i[src],dst,e_info))
                is_linked[buf_o2i[src]] = True
                is_linked[dst] = True
            else:
                new_edges.append((src,dst,e_info))
                is_linked[src] = True
                is_linked[dst] = True
        self.edges = new_edges



        # print(self.nodes)
        # for e in self.edges:
        #     print(e)

        # construct the graph
        src_nodes, dst_nodes = [[],[]],[[],[]]
        graph_info = {}
        node2nid = {}
        nid2node = {}
        nodes_type,nodes_delay,nodes_name, POs_label = [],[],[],[]
        is_po,is_pi= [],[]


        is_module = []
        # print(self.po_labels)
        # print(self.nodes.get('do_7_b53',None))
        # print(self.nodes.get('do_7_b53[0]',None))
        for node,node_info in self.nodes.items():
            if is_linked.get(node,None) is None:
                continue
            nid = len(node2nid)
            node2nid[node] = nid
            nid2node[node2nid[node]] = node
            nodes_name.append((node,node_info.get('nickname',None)))
            nodes_type.append(node_info['ntype'])
            is_module.append(node_info['is_module'])
            # set the PI delay
            #nodes_delay.append(self.pi_delay.get(node,0))
            if self.pi_delay.get(node,None) is not None:
                is_pi.append(1)
            else:
                is_pi.append(0)

            # set the PO label
            if node_info['is_po']:
                nickname = node_info.get('nickname',None)
                # print(node,nickname)
                if nickname is not None:
                    node = nickname
                if self.po_labels.get(node,None) is not None:
                    is_po.append(1)
                else:
                    is_po.append(0)
                    #POs_label.append(self.po_labels[node])
            else:
                is_po.append(0)

        # print({nodes_name[i]:nodes_delay[i] for i in range(len(nodes_name))})

        # get the src_node list and dst_node lsit

        bit_position = []
        for eid, (src, dst, edict) in enumerate(self.edges):
            edge_set_idx = is_module[node2nid[dst]]

            if node2nid.get(src,None) is not None:
                src_nodes[edge_set_idx].append(node2nid[src])
                dst_nodes[edge_set_idx].append(node2nid[dst])

                if edge_set_idx==1:
                    bit_position.append(edict['bit_position'])

                # print(edge_set_idx,node2nid[src],node2nid[dst])

        # print(self.nodes)
        #print(len(src_nodes[0]),len(src_nodes[1]))

        graph = dgl.heterograph(
        {('node', 'intra_module', 'node'): (th.tensor(src_nodes[1]), th.tensor(dst_nodes[1])),
         ('node', 'intra_gate', 'node'): (th.tensor(src_nodes[0]), th.tensor(dst_nodes[0]))
         }
        )
        graph.ndata['is_po'] = th.tensor(is_po)
        graph.ndata['is_pi'] = th.tensor(is_pi)
        graph.ndata['is_module'] = th.tensor(is_module)
        graph.edges['intra_module'].data['bit_position'] = th.tensor(bit_position, dtype=th.float)

        print('\t pre-filter: #node:{}, #edges:{}, {}'.format(graph.number_of_nodes(),
                                                               graph.number_of_edges('intra_gate'),
                                                               graph.number_of_edges('intra_module')))
        remain_nodes,remove_nodes = graph_filter(graph)
        remain_nodes = remain_nodes.numpy().tolist()
        graph.remove_nodes(remove_nodes)
        nodes_type = filter_list(nodes_type,remain_nodes)
        nodes_name = filter_list(nodes_name, remain_nodes)

        nodes_list = th.tensor(range(graph.number_of_nodes()))
        PIs_nid = nodes_list[graph.ndata['is_pi']==1].numpy().tolist()
        PIs_name = [nodes_name[n][0] for n in PIs_nid]
        POs_nid = nodes_list[graph.ndata['is_po'] == 1].numpy().tolist()
        POs_name = []
        for n in POs_nid:
            if nodes_name[n][1] is not None:
                POs_name.append(nodes_name[n][1])
            else:
                POs_name.append(nodes_name[n][0])

        topo = gen_topo(graph)
        PO2level = {}
        for l, nodes in enumerate(gen_topo(graph)):
            for n in nodes.numpy().tolist():
                if n in POs_nid:
                    PO2level[n] = l
        POs_level = [PO2level[n] for n in POs_nid]

        graph_info['topo'] = topo
        graph_info['ntype'] = nodes_type
        graph_info['nodes_name'] = nodes_name
        #graph_info['POs'] = POs
        #graph_info['POs_label'] = th.tensor(POs_label,dtype=th.float)
        graph_info['POs_level_max'] = th.tensor(POs_level,dtype=th.float)
        graph_info['POs_name'] = POs_name
        graph_info['PIs_name'] = PIs_name
        graph_info['case_name'] = self.case_name
        graph_info['design_name'] = self.design_name


        # POs_level_min = []     # record the shortest path length from an PI to any PI
        # for po in POs:
        #     l = 0
        #     # print(po)
        #     predecessors = graph.predecessors(po).numpy().tolist()
        #     # predecessors_name = [nid2node[n] for n in predecessors]
        #     while len(predecessors) !=0:
        #         l +=1
        #         new_predecessors = []
        #         for p in predecessors:
        #             pre_preds = graph.predecessors(p).numpy().tolist()
        #             if len(pre_preds)==0:
        #                 break
        #             new_predecessors.extend(pre_preds)
        #         # predecessors_name = [nid2node[n] for n in new_predecessors]
        #         predecessors = new_predecessors
        #         # print('\t', predecessors_name)
        #     POs_level_min.append(l)
        # graph_info['POs_level_min'] = th.tensor(POs_level_min,dtype=th.float)

        # print('\t',graph)
        print('\t post-filter: #node:{}, #edges:{}, {}'.format(graph.number_of_nodes(),graph.number_of_edges('intra_gate'),graph.number_of_edges('intra_module')))
        # print('\t',graph_info)
        #print('\t POs: max levels={}'.format(POs_level))
        #print('\t      min_levels={}'.format(POs_level_min))

        return graph, graph_info

    def parse(self):
        self.pi_delay,self.po_labels = parse_golden(os.path.join(self.design_path,'golden.txt'))
        if self.pi_delay is None:
            return None,None
        graph, graph_info = self.parse_verilog()
        return graph,graph_info


def find_precessors(graph,graph_info,node):

    node_names = graph_info['nodes_name']
    pre = list(graph.predecessors(node))

    cur = []
    print(node_names[node])
    print([node_names[n] for n in pre])
    while len(pre)!=0:
        for nid in pre:
            cur.extend(graph.predecessors(nid))
        pre = cur
        cur = []
        print([node_names[n] for n in pre])

def parse_golden(file_path):



    with open(file_path, 'r') as f:
        content = f.read()
    pi_delay = {}
    po_labels = {}
    if 'pin to pin level synthesised' not in content:
        return None,None
    pi_content,po_content = content.split('// pin to pin level synthesised\n')

    for line in pi_content.split('\n'):
        if '//' in line or len(line)==0:
            continue
        pi,delay = line.split(' ')
        pi_delay[pi] = int(delay)
    for line in po_content.split('\n'):
        if '//' in line or len(line)==0:
            continue

        po, label = line.split(' ')
        po = po.replace(',','')
        po_labels[po] = int(label)
    # print(self.pi_delay)
    # print(self.po_labels)
    return pi_delay,po_labels




def main():
    dataset = []
    num = 0

    for subdir in ['']:
    #for subdir in os.listdir(rawdata_path):
        subdir_path = os.path.join(rawdata_path,subdir)
        design2idx = {}
        for design in os.listdir(subdir_path):
            # if '00320' in design:
            #     continue

            design_name = design[:design.rfind('_')]
            #print(subdir_path,design)
            design_idx,case_idx = design.split('_')[-2:]

            design_idx = int(design_idx)
            # if design_idx>10:
            #     continue
            case_idx = int(case_idx)
            design2idx[design_name] = design2idx.get(design_name,[])
            design2idx[design_name].append(case_idx)


        for design_name, case_indexs in design2idx.items():
            #if '370' not in design_name: continue
            #if '399' not in design_name:
            #    print(design_name)
            #    continue
            if 0 not in case_indexs:
                print("skip {}, due to the miss of idx 0".format(design_name))

            base_pi_delay, base_po_labels = parse_golden(os.path.join(subdir_path, '{}_{}'.format(design_name,0),'golden.txt'))
            for p,d in base_pi_delay.items():
                assert d == 0, print("base case with non-zero input delay: {} {}".format(p,d))


            print("Processing {}/{}, #{}".format(subdir, design_name, num))
            design_dir = os.path.join(subdir_path, '{}_{}'.format(design_name,0))
            parser = Parser(subdir, design_dir)

            graph, graph_info = parser.parse()
            if graph is None:
                continue

            case_indexs = sorted(case_indexs)

            graph_info['delay-label_pairs'] = []
            base_labels = {}
            for idx in case_indexs:
                # if idx==0:
                #     continue
                golden_file_path = os.path.join(subdir_path, '{}_{}'.format(design_name,idx),'golden.txt')
                pi_delay,po_labels = parse_golden(golden_file_path)
                #print(design_name,idx,po_labels)
                if len(po_labels)!=len(base_po_labels):
                    continue
                po_labels_residual = {p: d-base_po_labels[p] for (p,d) in po_labels.items()}
                #print('\t',po_labels)


                temp_nodenames = []
                cur_pis = []
                PIs_delay, POs_label,POs_label_residual, POs = [], [],[], [ ]
                if pi_delay is None:
                    continue

                for node in graph_info["PIs_name"]:
                    PIs_delay.append(pi_delay[node])

                for node in graph_info['POs_name']:
                    POs_label.append(po_labels[node])
                    POs_label_residual.append(po_labels_residual[node])

                #print(idx,len(PIs_delay),th.sum(graph.ndata['is_pi']).item(),len(POs_label),th.sum(graph.ndata['is_po']).item())
                assert len(PIs_delay) == th.sum(graph.ndata['is_pi']).item() and len(POs_label) == th.sum(graph.ndata['is_po']).item()
                graph_info['delay-label_pairs'].append((PIs_delay, POs_label,POs_label_residual))



            POs_base_label = []
            for node in graph_info['POs_name']:
                POs_base_label.append(base_po_labels[node])
            graph_info['base_po_labels'] = POs_base_label

            if len(graph_info['delay-label_pairs'])==0:
                continue
            num += 1
            if graph is not None:
                dataset.append((graph,graph_info))
            # print(graph.ndata['is_pi'])
            # print(graph_info['delay-label_pairs'])
        # exit()
    if not os.path.exists(ntype_file):
        with open(ntype_file,'wb') as f:
            pickle.dump((ntype2id,ntype2id_gate,ntype2id_module),f)
    print('ntypes:',ntype2id,ntype2id_gate,ntype2id_module)

    final_dataset = []
    for graph, graph_info in dataset:
        is_module = graph.ndata['is_module'].numpy().tolist()
        ntype_onehot = th.zeros((graph.number_of_nodes(), len(ntype2id)), dtype=th.float)
        ntype_onehot_module = th.zeros((graph.number_of_nodes(), len(ntype2id_module)), dtype=th.float)
        ntype_onehot_gate = th.zeros((graph.number_of_nodes(), len(ntype2id_gate)), dtype=th.float)

        for nid, type in enumerate(graph_info['ntype']):
            ntype_onehot[nid][ntype2id[type]] = 1
            if is_module[nid] == 1:
                ntype_onehot_module[nid][ntype2id_module[type]] = 1
            else:
                ntype_onehot_gate[nid][ntype2id_gate[type]] = 1
        graph.ndata['ntype'] = ntype_onehot
        graph.ndata['ntype_module'] = ntype_onehot_module
        graph.ndata['ntype_gate'] = ntype_onehot_gate
        final_dataset.append((graph,graph_info))

    dataset = final_dataset
    shuffle(dataset)
    split_ratio = [0.7, 0.1, 0.2]
    num_designs = len(dataset)
    data_train = dataset[:int(0.7 * num_designs)]
    data_val = dataset[int(0.7 * num_designs):int(0.8 * num_designs)]
    data_test = dataset[int(0.8 * num_designs):]
    print('#train:{}, #val:{}, #test:{}'.format(len(data_train),len(data_val),len(data_test)))
    with open(os.path.join(data_savepath,'data_train.pkl'),'wb') as f:
        pickle.dump(data_train,f)
    with open(os.path.join(data_savepath,'data_val.pkl'),'wb') as f:
        pickle.dump(data_val,f)
    with open(os.path.join(data_savepath,'data_test.pkl'),'wb') as f:
        pickle.dump(data_test,f)
    #print(len(dataset))
    # with open(os.path.join(data_savepath,'data.pkl'),'wb') as f:
    #     pickle.dump(dataset,f)
    # with open(os.path.join(data_savepath,'graph.pkl'),'wb') as f:
    #     pickle.dump(final_dataset,f)

if __name__ == "__main__":
    stdout_f = os.path.join(data_savepath,'stdout.log')
    stderr_f = os.path.join(data_savepath, 'stderr.log')

    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
        main()
