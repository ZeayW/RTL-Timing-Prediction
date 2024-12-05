import torch as th
import dgl
import os
import pickle
from options import get_options
from random import shuffle
import tee
from utils import *

options = get_options()
rawdata_path = options.rawdata_path
data_savepath = options.data_savepath
os.makedirs(data_savepath,exist_ok=True)

ntype_file = os.path.join(data_savepath,'ntype2id.pkl')

ntype2id = {'input':0, "1'b0":1, "1'b1":2}
ntype2id_gate = {'input':0, "1'b0":1, "1'b1":2}
ntype2id_module = {}

def get_nodename(nodes_name,nid,is_po):
    if is_po==1 and nodes_name[nid][1] is not None:
        return nodes_name[nid][1]
    else:
        return nodes_name[nid][0]


class Parser:
    def __init__(self,design_dir):
        self.design_name = os.path.split(design_dir)[-1]
        self.design_path = design_dir
        self.wires_width = {}
        self.const0_index = 0
        self.const1_index = 0
        self.nodes = {}
        self.edges = [] # edges: List[Tuple[str, str, Dict]] = []  # a list of (src, dst, {key: value})
        self.pi_delay = {}
        self.po_labels = {}

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
        file_path = os.path.join(self.design_path,'{}_case.v'.format(self.design_name))

        with open(file_path, 'r') as f:
            content = f.read()
        content = content[content.find('input'):]

        buf_o2i = {}
        buf_i2o = {}

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
                        # only when the output of a buf is a PO, then we will reserve this buf output node
                        #   and we need to link the inputs to the buf input to the buf output node
                        if self.nodes[fanout_node]['is_po']:
                            buf_i2o[fanin_node] = fanout_node
                            self.nodes[fanout_node]['ntype'] = self.nodes[fanin_node]['ntype']

                        else:
                            buf_o2i[fanout_node] = fanin_node
                            self.nodes[fanin_node]['nicknames'] = self.nodes[fanin_node].get('nicknames',[])
                            self.nodes[fanin_node]['nicknames'].append(fanout_node)
                            self.nodes[fanin_node]['is_po'] = self.nodes[fanout_node]['is_po']
                            self.nodes[fanout_node]['ntype'] = None

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

        is_linked = {}
        new_edges = []
        for src,dst, e_info in self.edges:
            new_src = buf_o2i.get(src,src)
            new_dst = buf_i2o.get(dst,dst)
            if new_dst!=dst:
                self.nodes[new_dst]['ntype'] = self.nodes[dst]['ntype']
                self.nodes[new_dst]['is_module'] = self.nodes[dst]['is_module']

            new_edges.append((new_src, new_dst, e_info))
            is_linked[new_src] = True
            is_linked[new_dst] = True
        self.edges = new_edges

        self.nodes = {n: self.nodes[n] for n in self.nodes.keys() if self.nodes[n]['ntype'] not in ['wire', None]}

        # construct the graph
        src_nodes, dst_nodes = [[],[]],[[],[]]
        graph_info = {}
        node2nid = {}
        nid2node = {}
        nodes_type,nodes_delay,nodes_name, POs_label = [],[],[],[]
        is_po,is_pi= [],[]
        is_module = []
        for node,node_info in self.nodes.items():
            if not node_info['is_po'] and is_linked.get(node,None) is None:
                continue
            nid = len(node2nid)
            node2nid[node] = nid
            nid2node[node2nid[node]] = node

            nodes_name.append((node,node_info.get('nicknames',None)))
            nodes_type.append(node_info['ntype'])
            is_module.append(node_info['is_module'])
            # set the PI delay
            #nodes_delay.append(self.pi_delay.get(node,0))
            if self.pi_delay.get(node,None) is not None:
                is_pi.append(1)
            else:
                is_pi.append(0)

            # set the PO label
            flag_po = False
            if node_info['is_po']:
                nicknames = node_info.get('nicknames',None)
                if nicknames is None:
                    nicknames  = [node]
                for nickname in nicknames:
                    if self.po_labels.get(nickname, None) is not None:
                        flag_po = True
                        if len(nicknames)!=1:
                            node_info['nicknames'] = [nickname]
                            node_info['nicknames'].extend(nicknames)
                        break
            if flag_po:
                is_po.append(1)
            else:
                is_po.append(0)

        # get the src_node list and dst_node lsit
        bit_position = []
        for eid, (src, dst, edict) in enumerate(self.edges):
            edge_set_idx = is_module[node2nid[dst]]

            if node2nid.get(src,None) is not None:
                src_nodes[edge_set_idx].append(node2nid[src])
                dst_nodes[edge_set_idx].append(node2nid[dst])

                if edge_set_idx==1:
                    bit_position.append(edict['bit_position'])

        graph = dgl.heterograph(
        {('node', 'intra_module', 'node'): (th.tensor(src_nodes[1]), th.tensor(dst_nodes[1])),
         ('node', 'intra_gate', 'node'): (th.tensor(src_nodes[0]), th.tensor(dst_nodes[0]))
         },num_nodes_dict={'node':len(node2nid)}
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

        is_module = filter_list(is_module,remain_nodes)
        nodes_type = filter_list(nodes_type,remain_nodes)
        nodes_name = filter_list(nodes_name, remain_nodes)

        nodes_list = th.tensor(range(graph.number_of_nodes()))
        PIs_nid = nodes_list[graph.ndata['is_pi']==1].numpy().tolist()
        PIs_name = [nodes_name[n][0] for n in PIs_nid]
        POs_nid = nodes_list[graph.ndata['is_po'] == 1].numpy().tolist()
        POs_name = []
        for n in POs_nid:
            if nodes_name[n][1] is not None:
                POs_name.append(nodes_name[n][1][0])
            else:
                POs_name.append(nodes_name[n][0])

        nname2nid = {}
        for nid,nname in enumerate(nodes_name):
            nname2nid[nname[0]] = nid
            if nname[1] is not None:
                for nm in nname[1]:
                    nname2nid[nm] = nid

        # print(nname2nid)

        topo = gen_topo(graph)
        PO2level = {}
        for l, nodes in enumerate(gen_topo(graph)):
            for n in nodes.numpy().tolist():
                if n in POs_nid:
                    PO2level[n] = l
        POs_level = [PO2level[n] for n in POs_nid]

        remain_pos_idx = []

        for i,level in enumerate(POs_level):
            nid = POs_nid[i]
            PO_name = POs_name[i]
            PO_label = self.po_labels[PO_name]
            if PO_label==0 and level>=2:
                print('\t removing PO:',PO_name,PO_label,level)
                graph.ndata['is_po'][nid] = 0
            else:
                remain_pos_idx.append(i)

        POs_level = filter_list(POs_level,remain_pos_idx)
        POs_name = filter_list(POs_name, remain_pos_idx)
        POs_nid = filter_list(POs_nid, remain_pos_idx)



        graph_info['topo'] = topo
        graph_info['ntype'] = nodes_type
        graph_info['nodes_name'] = nodes_name
        graph_info['nname2nid'] = nname2nid
        #graph_info['POs'] = POs
        #graph_info['POs_label'] = th.tensor(POs_label,dtype=th.float)
        graph_info['POs_level_max'] = th.tensor(POs_level,dtype=th.float)
        graph_info['POs_name'] = POs_name
        graph_info['PIs_name'] = PIs_name
        graph_info['design_name'] = self.design_name

        #print(graph_info['POs_name'],len(graph_info['POs_name']),len(graph.ndata['is_po'][graph.ndata['is_po']==1]))
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

        print('\t post-filter: #node:{}, #edges:{}, {}'.format(graph.number_of_nodes(),graph.number_of_edges('intra_gate'),graph.number_of_edges('intra_module')))
        # print('\t',graph_info)
        #print('\t POs: max levels={}'.format(POs_level))
        #print('\t      min_levels={}'.format(POs_level_min))

        return graph, graph_info

    def parse(self):
        self.pi_delay,self.po_labels,_ = parse_golden(os.path.join(self.design_path,'golden_0.txt'))
        for p, d in self.pi_delay.items():
            assert d == 0, print("base case with non-zero input delay: {} {}".format(p, d))
        if self.pi_delay is None:
            return None,None
        graph, graph_info = self.parse_verilog()
        graph_info['base_po_labels'] = self.po_labels

        return graph,graph_info

def parse_golden(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    pi_delay = {}
    po_labels = {}
    po_criticalPIs = {}
    if 'pin to pin level synthesised' not in content:
        return None,None

    pi_content,po_content = content.split('// pin to pin level synthesised\n')[:2]

    for line in pi_content.split('\n'):
        if '//' in line or len(line)==0:
            continue
        pi,delay = line.split(' ')
        pi_delay[pi] = int(delay)
    for line in po_content.split('\n'):
        if '//' in line or len(line)==0:
            continue

        po, pi,label = line.split(' ')
        po = po.replace(',','')
        pi = pi.replace(',','')
        po_criticalPIs[po] = po_criticalPIs.get(po,[])
        po_criticalPIs[po].append(pi)
        po_labels[po] = int(label)
    # print(self.pi_delay)
    # print(self.po_labels)
    return pi_delay,po_labels,po_criticalPIs




def main():
    dataset = []
    num = 0

    for subdir in ['']:
    #for subdir in os.listdir(rawdata_path):
        subdir_path = os.path.join(rawdata_path,subdir)
        design2idx = {}
        for design in os.listdir(subdir_path):
            # if '00072' not in design:
            #     continue
            # if '00097' not in design:
            #     continue
            design_dir = os.path.join(subdir_path,design)
            if not os.path.isdir(design_dir):
                continue
            print("-----Parsing {}-----".format(design))
            parser = Parser(design_dir)
            graph, graph_info = parser.parse()
            if graph is None:
                continue

            label_files = [f for f in os.listdir(design_dir) if f.startswith('gold')]
            case_indexs = [int(f.split('_')[-1].split('.')[0]) for f in label_files]
            case_indexs = sorted(case_indexs)

            graph_info['delay-label_pairs'] = []
            base_labels = {}
            for idx in case_indexs:
                # if idx==0:
                #     continue
                golden_file_path = os.path.join(design_dir, 'golden_{}.txt'.format(idx))

                pi_delay,po_labels,po_criticalPIs = parse_golden(golden_file_path)
                #print(design_name,idx,po_labels)
                if len(po_labels)!=len(graph_info['base_po_labels']):
                    continue

                po_labels_residual = {p: d-graph_info['base_po_labels'][p] for (p,d) in po_labels.items()}
                PIs_delay, POs_label,POs_label_residual, POs = [], [],[], [ ]
                if pi_delay is None:
                    continue

                for node in graph_info["PIs_name"]:
                    PIs_delay.append(pi_delay[node])

                for node in graph_info['POs_name']:
                    POs_label.append(po_labels[node])
                    POs_label_residual.append(po_labels_residual[node])

                pi2po_edges = ([],[])
                for po, critical_pis in po_criticalPIs.items():
                    po_nid = graph_info['nname2nid'][po]
                    critical_pi_nids = [graph_info['nname2nid'][pi] for pi in critical_pis]
                    pi2po_edges[0].extend(critical_pi_nids)
                    pi2po_edges[1].extend([po_nid]*len(critical_pi_nids))
                # nodes_delay = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                # nodes_delay[graph.ndata['is_pi'] == 1] = th.tensor(PIs_delay, dtype=th.float).unsqueeze(1)
                # graph.ndata['delay'] = nodes_delay
                # pi2po_edges,edges_weight = get_pi2po_edges(graph,graph_info)
                #

                #print(idx,len(PIs_delay),th.sum(graph.ndata['is_pi']).item(),len(POs_label),th.sum(graph.ndata['is_po']).item())
                assert len(PIs_delay) == th.sum(graph.ndata['is_pi']).item() and len(POs_label) == th.sum(graph.ndata['is_po']).item()
                graph_info['delay-label_pairs'].append((PIs_delay, POs_label,POs_label_residual,pi2po_edges))



            POs_base_label = []
            for node in graph_info['POs_name']:
                POs_base_label.append(graph_info['base_po_labels'][node])
            graph_info['base_po_labels'] = POs_base_label

            if len(graph_info['delay-label_pairs'])<=1:
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
