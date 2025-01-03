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

# outputs:
#   0: constant 1'b0
#   1: constant 1'b1
#   2: buf
#   3: not
def func_xor(inputs):
    assert len(inputs) == 2
    sum_i = sum(inputs)
    if sum_i==-2:
        return 2
    elif sum_i==-1:
        return 3
    else:
        return sum_i%2
def func_xnor(inputs):
    assert len(inputs) == 2
    sum_i =sum(inputs)
    if sum_i==-2:
        return 3
    elif sum_i==-1:
        return 2
    else:
        return (sum_i+1)%2

def func_or(inputs):
    assert len(inputs) == 2
    sum_i = sum(inputs)
    if sum_i==-2:
        return 2
    elif sum_i==-1:
        return 1
    else:
        return 1 if sum_i>=1 else 0
def func_and(inputs):
    assert len(inputs) == 2
    sum_i = sum(inputs)
    if sum_i==-2:
        return 0
    elif sum_i==-1:
        return 2
    else:
        return 1 if sum_i==2 else 0

def func_not(inputs):
    assert len(inputs)==1
    return (inputs[0]+1)%2

def func_buf(inputs):
    assert len(inputs)==1
    return inputs[0]

gate_func_map = {
    'xor':func_xor,
    'xnor':func_xnor,
    'and':func_and,
    'or':func_or,
    'not':func_not,
    'buf':func_buf
}


def edges_srcLevel_rating(graph,nodes_level,etype):
    src_level = {}
    dsts = set()
    intra_gate_edges = graph.edges('all', etype=etype)[:2]
    intra_gate_edges = list(zip(intra_gate_edges[0].numpy().tolist(), intra_gate_edges[1].numpy().tolist()))
    edges_level_rating = th.zeros((len(intra_gate_edges), 1), dtype=th.float)
    for i, (src, dst) in enumerate(intra_gate_edges):
        dsts.add(dst)
        src_level[dst] = src_level.get(dst, [])
        src_level[dst].append((src, nodes_level[src], i))
    for dst in dsts:
        cur_src_level = src_level[dst]
        cur_src_level.sort(key=lambda x: x[1])
        cur_src_level.reverse()
        rating = 0
        pre_level = -1
        for j, (src, level, eid) in enumerate(cur_src_level):
            if level!=pre_level:
                rating += 1
            pre_level = level
            edges_level_rating[eid] = rating

    return edges_level_rating

class Parser:
    def __init__(self,design_dir):
        self.design_name = os.path.split(design_dir)[-1]
        self.design_path = design_dir
        #print(self.design_name,design_dir)
        self.wires_width = {}
        self.const0_index = 0
        self.const1_index = 0
        self.nodes = {}
        self.edges = [] # edges: List[Tuple[str, str, Dict]] = []  # a list of (src, dst, {key: value})
        self.fo2fi = {}
        self.pi_delay = {}
        self.po_labels = {}
        self.nicknames = {}
        #self.buf_types = ['buf','not']
        self.buf_types = ['buf']

    def is_constant(self,node):
        if "1'b0" in node or self.nodes.get(node,{}).get('ntype')=="1'b0":
            return 0
        elif "1'b1" in node or self.nodes.get(node,{}).get('ntype')=="1'b1":
            return 1
        else:
            return -2

    def get_moduleIOnodes(self,sentence):
        io_wires = sentence[sentence.find('(') + 1:].strip()
        io_wires = io_wires.split('),')
        io_wires = [p.replace(' ', '') for p in io_wires]
        io_wires = {p[1:p.find('(')]: p[p.find('(') + 1:].strip().replace(')', '') for p in io_wires}
        assert io_wires.get('o', None) is not None

        io_nodes = {p: self.parse_wire(w) for p, w in io_wires.items()}

        return io_nodes

    #.i0({3'b000,do_2_b5_n,do_2_b5_n1,do_2_b5_n2,do_2_b5_n3,do_2_b5_n4}),
    # parse the given wire, and return the list of single-bit pins
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
                    if options.flag_split01:
                        node = "1'b0_{}".format(self.const0_index)
                    else:
                        node = "1'b0"
                    self.nodes[node] = {'ntype': "1'b0", 'is_po': False,'is_module':0}
                    res.append(node)
                elif v == "1":
                    self.const1_index += 1
                    if options.flag_split01:
                        node = "1'b1_{}".format(self.const1_index)
                    else:
                        node = "1'b1"
                    self.nodes[node] = {'ntype': "1'b1", 'is_po': False,'is_module':0}
                    res.append(node)
                else:
                    assert False, "{}".format(wire)
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

        # if 'lt2023' in content and 'lt2024' not in content:
        #     error_region = content[content.find('lt2023'):content.find('lt2025')]
        #     error_region = error_region[:error_region.rfind('.o')]
        #     fixed_region = error_region.replace(';\n',';\n  lt_u1_u1 lt2024 (\n')
        #     #print(fixed_region)
        #     content = content.replace(error_region,fixed_region)
        #     with open(file_path, 'w') as f:
        #         f.write(content)
        #     print('---fix')
            # return None, {}

        buf_o2i, buf_i2o = {}, {}
        fo2fi_bit_position = {}


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
                # get the gate type
                gate = sentence[:sentence.find('(')]
                if len(gate.strip().split(' '))==1:
                    print(sentence)
                gate_type, gate_name = gate.strip().split(' ')


                # deal with multiplexer, whose width may be larger than 1
                if 'mux' in gate_type.lower():
                    gate_type = 'mux'
                    io_nodes = self.get_moduleIOnodes(sentence)
                    assert len(io_nodes) == 4

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
                        self.fo2fi[n] = ('mux',gate_name,[])
                    # add the edges between fanin nodes and fanout nodes
                    for port, fanin_nodes in io_nodes.items():
                        if port == 'o':
                            continue
                        # for port i0,i1: link fi_i[j] with fo[j]
                        elif 'i' in port:
                            for i,fanout_node in enumerate(fanout_nodes):
                                fo2fi[fanout_node].append(fanin_nodes[i])
                                self.fo2fi[fanout_node][2].append(fanin_nodes[i])
                        # for port sel: link all fi_s with fo[j]
                        elif port=='sel':
                            for i, fanout_node in enumerate(fanout_nodes):
                                fo2fi[fanout_node].extend(fanin_nodes)
                                self.fo2fi[fanout_node][2].extend(fanin_nodes)
                        else:
                            assert False
                # deal with arithmetic blocks, whose width may be larger than 1
                elif '.' in sentence:
                    gate_type = gate_type.split('_')[0]
                    io_nodes = self.get_moduleIOnodes(sentence)

                    # set the node information for the fanout nodes
                    fanout_nodes = io_nodes['o']
                    fanout_nodes.reverse()
                    # if gate_type in ['decoder','encoder'] and len(io_nodes['i'])>=32:
                    #     gate_type = 'input'
                    #     for n in fanout_nodes:
                    #         if self.nodes.get(n, None) is None and 'open' in n:
                    #             self.nodes[n] = {'ntype': gate_type, 'is_po': False, 'is_module': 0}
                    #         else:
                    #             self.nodes[n]['ntype'] = gate_type
                    #             self.nodes[n]['is_module'] = 0
                    #     continue

                    for n in fanout_nodes:
                        if self.nodes.get(n, None) is None and 'open' in n:
                            self.nodes[n] = {'ntype': gate_type, 'is_po': False,'is_module':1}
                        else:
                            self.nodes[n]['ntype'] = gate_type
                            self.nodes[n]['is_module'] = 1

                        ntype2id[gate_type] = ntype2id.get(gate_type, len(ntype2id))
                        ntype2id_module[gate_type] = ntype2id_module.get(gate_type, len(ntype2id_module))
                        fo2fi[n] = []
                        self.fo2fi[n] = ('module', gate_name,[])


                    width = 0
                    # add the edges between fanin nodes and fanout nodes
                    for port, fanin_nodes in io_nodes.items():
                        for idx,fi in enumerate(fanin_nodes):
                            fo2fi_bit_position[(gate_name,fi)] = len(fanin_nodes) - idx

                            #fanins_bit_position[fi] = len(fanin_nodes) - idx
                        fanin_nodes.reverse()

                        if port == 'o':
                            continue
                        # for port i0,i1: link fi_i[j...0] with fo[j]
                        elif port.startswith('i') and len(fanout_nodes)!=1 and gate_type not in ['encoder','decoder']:

                            for i, fanout_node in enumerate(fanout_nodes):
                                fo2fi[fanout_node].extend(fanin_nodes[:i+1])
                                self.fo2fi[fanout_node][2].extend(fanin_nodes[:i+1])
                                self.nodes[fanout_node]['width'] = i+1

                        # for port sel or one output modules, e.g., eq, lt: link all fi_s with each fo
                        else:
                            for i, fanout_node in enumerate(fanout_nodes):
                                fo2fi[fanout_node].extend(fanin_nodes)
                                self.fo2fi[fanout_node][2].extend(fanin_nodes)
                                self.nodes[fanout_node]['width'] = len(fanin_nodes)

                        #width = max(len(fanin_nodes), width)
                # deal with other one-output gates, e.g., or, and...
                else:
                    # get the paramater list
                    io_wires = sentence[sentence.find('(') + 1:]
                    io_wires = io_wires.replace(')', '').split(',')
                    io_wires = [p.replace(' ','') for p in io_wires]
                    fanout_node = io_wires[0]  # fanout is the first parameter

                    fanin_nodes = [self.parse_wire(w)[0] for w in io_wires[1:]]

                    if gate_type not in self.buf_types:
                        ntype2id[gate_type] = ntype2id.get(gate_type, len(ntype2id))
                        ntype2id_gate[gate_type] = ntype2id_gate.get(gate_type, len(ntype2id_gate))
                    fo2fi[fanout_node] = fanin_nodes
                    self.fo2fi[fanout_node] = ('gate', gate_name,fanin_nodes)
                    self.nodes[fanout_node]['ntype'] = gate_type



        # deal with the constant inputs (1'b0/1) iteratively
        flag_stop = False
        while not flag_stop:
            num_removed_constant = 0
            new_fo2fi = {}
            for fanout, (fanout_type,cell_name,fanins) in self.fo2fi.items():
                # simplify the logic gates with constant input (1'b0/1'b1)
                if fanout_type == 'gate':
                    gate_type = self.nodes[fanout]['ntype']
                    # check if the inputs contain any constant value
                    flag_input_constant = [self.is_constant(n) for n in fanins]
                    if 0 not in flag_input_constant and 1 not in flag_input_constant:
                        new_fo2fi[fanout] = (fanout_type,cell_name, fanins)
                        continue
                    num_removed_constant += 1
                    # do the simplification based on pre-defined calculation
                    gate_func = gate_func_map[gate_type]
                    flag_output = gate_func(flag_input_constant)
                    if flag_output == 0:
                        gate_type = "1'b0"
                    elif flag_output == 1:
                        gate_type = "1'b1"
                    elif flag_output == 2:
                        gate_type = "buf"
                    elif flag_output == 3:
                        gate_type = "not"
                    # update the gate type
                    self.nodes[fanout]['ntype'] = gate_type

                    if gate_type in ["1'b0","1'b1"]:
                        continue

                    # record the new edges
                    fanins = [n for i, n in enumerate(fanins) if flag_input_constant[i] == -2]
                    assert gate_type not in ['buf','not'] or len(fanins)!=0, "fanout {} with no fanin".format(fanout)
                    if len(fanins) != 0:
                        new_fo2fi[fanout] = (fanout_type, cell_name,fanins)
                else:
                    new_fo2fi[fanout] = (fanout_type,cell_name,fanins)

            self.fo2fi = new_fo2fi
            flag_stop = num_removed_constant==0

        # deal with the buffers/NOT gates
        #   record the input-output and output-input pair of buffer/NOT
        for fanout, (fanout_type,_, fanins) in self.fo2fi.items():
            gate_type = self.nodes[fanout]['ntype']
            if gate_type in self.buf_types:
                assert len(fanins)==1
                fanin = fanins[0]
                buf_o2i[fanout] = fanin
                buf_i2o[fanin] = fanout



        # get the edges and check whether each node is connected or not
        is_linked = {}
        visited_po = {}
        for fanout, (fanout_type, cell_name, fanins) in self.fo2fi.items():

            dst = fanout
            src_list = []

            # add edges from fanin to fanout, consdiering replacement of buffer
            if self.nodes[fanout]['ntype'] not in self.buf_types:
                for fanin in fanins:
                    src = fanin
                    num_inv = 0
                    # for each fanin, recusively backtrace to predecessor until meet non-buffer node
                    while buf_o2i.get(src, None) is not None:
                        src = buf_o2i[src]
                        if self.nodes[src]['ntype'] == 'not':
                            num_inv += 1
                    is_inv = num_inv%2
                    bit_position = fo2fi_bit_position.get((cell_name,fanin),None)
                    # if self.nodes[fanout]['ntype'] in ['ne','eq']:
                    #     bit_position = 0
                    # if bit_position is not None and bit_position>100:
                    #     print(src,dst)

                    src_list.append((src,bit_position,is_inv))
                    self.edges.append(
                        (src, dst, {'bit_position': bit_position,'is_inv':is_inv})
                    )

                    is_linked[src] = True
                    is_linked[dst] = True

            # deal with the special condition that the buf output is PO while buf input is PI
            else:
                assert len(fanins) == 1
                src = fanins[0]
                while buf_o2i.get(src, None) is not None:
                    src = buf_o2i[src]

                if self.nodes[fanout]['is_po'] and self.nodes[src]['ntype'] in ["input", "1'b0", "1'b1"]:
                    assert not visited_po.get(fanout,False)
                    self.nodes[fanout]['ntype'] = self.nodes[src]['ntype']
                    self.nodes[fanout]['width'] = self.nodes[src].get('width',0)
                    self.nicknames[fanout] = src
                    visited_po[fanout] = True
                continue

            # when a buf node is also a PO
            # we will duplicate the buf input node to reserce the PO
            num_inv = 0
            # recusively visit successors of fanout node until meet non-buffer node
            while buf_i2o.get(dst,None) is not None:
                dst = buf_i2o[dst]
                if self.nodes[dst]['ntype']:
                    num_inv += 1
            # check whether the last visited node is PO
            if self.nodes[dst]['is_po']:
                assert not visited_po.get(dst, False)
                for (src, bit_pos,is_inv) in src_list:
                    is_inv = (is_inv+num_inv)%2
                    self.edges.append(
                        (src, dst, {'bit_position': bit_pos,'is_inv':is_inv})
                    )
                    is_linked[src] = True
                    is_linked[dst] = True

                self.nodes[dst]['ntype'] = self.nodes[fanout]['ntype']
                self.nodes[dst]['is_module'] = self.nodes[fanout]['is_module']


        self.nodes = {n: self.nodes[n] for n in self.nodes.keys() if self.nodes[n]['ntype'] not in ['wire', None]}

        # construct the graph
        src_nodes, dst_nodes = [[],[]],[[],[]]
        graph_info = {}
        node2nid = {}
        nid2node = {}
        nodes_type,nodes_value,nodes_delay,nodes_name, nodes_width,POs_label, = [],[],[],[],[],[]
        is_po,is_pi= [],[]
        is_module = []
        for node,node_info in self.nodes.items():
            if not node_info['is_po'] and is_linked.get(node,None) is None:
                continue
            nid = len(node2nid)
            node2nid[node] = nid
            nid2node[node2nid[node]] = node

            nodes_name.append(node)
            #nodes_name.append((node,node_info.get('nicknames',None)))
            nodes_type.append(node_info['ntype'])
            if node_info['ntype'] in self.buf_types or node_info['ntype']=='output':
                print("***",node,node_info)
                exit()

            is_module.append(node_info['is_module'])
            nodes_width.append(node_info.get('width',0))
            if node_info['ntype'] == "1'b0":
                nodes_value.append(0)
            elif node_info['ntype'] == "1'b1":
                nodes_value.append(1)
            else:
                nodes_value.append(2)

            # set the PI delay
            #nodes_delay.append(self.pi_delay.get(node,0))
            flag_pi, flag_po = 0,0
            if self.pi_delay.get(node,None) is not None:
                flag_pi = 1
            if node_info['is_po'] and self.po_labels.get(node, None) is not None:
                flag_po = 1
                if node_info['ntype'] in ["input","1'b0","1'b1"]:
                    flag_pi = 1
            is_pi.append(flag_pi)
            is_po.append(flag_po)

        nodes_valueOnehot = th.zeros((len(nodes_name),3),dtype=th.float)

        for i, v in enumerate(nodes_value):
            nodes_valueOnehot[i][v] = 1

        # get the src_node list and dst_node lsit
        bit_position = []
        is_inv = [[],[]]
        for eid, (src, dst, edict) in enumerate(self.edges):
            #print(src,dst)

            assert nodes_value[node2nid[dst]] not in [0,1]
            edge_set_idx = is_module[node2nid[dst]]

            if node2nid.get(src,None) is not None:
                src_nodes[edge_set_idx].append(node2nid[src])
                dst_nodes[edge_set_idx].append(node2nid[dst])
                is_inv[edge_set_idx].append(edict['is_inv'])

                if edge_set_idx==1:
                    bit_position.append(edict['bit_position'])

        #print(len(src_nodes[1]), len(src_nodes[0]))
        graph = dgl.heterograph(
        {('node', 'intra_module', 'node'): (th.tensor(src_nodes[1]), th.tensor(dst_nodes[1])),
         ('node', 'intra_gate', 'node'): (th.tensor(src_nodes[0]), th.tensor(dst_nodes[0]))
         },num_nodes_dict={'node':len(node2nid)}
        )

        graph.ndata['is_po'] = th.tensor(is_po)
        graph.ndata['is_pi'] = th.tensor(is_pi)
        graph.ndata['is_module'] = th.tensor(is_module)
        graph.ndata['width'] = th.tensor(nodes_width,dtype=th.float).unsqueeze(1)
        graph.ndata['value'] = nodes_valueOnehot
        graph.edges['intra_module'].data['bit_position'] = th.tensor(bit_position, dtype=th.float)
        graph.edges['intra_module'].data['is_inv'] = th.tensor(is_inv[1], dtype=th.float)
        graph.edges['intra_gate'].data['is_inv'] = th.tensor(is_inv[0], dtype=th.float)


        print('\t pre-filter: #node:{}, #edges:{}, {}'.format(graph.number_of_nodes(),
                                                               graph.number_of_edges('intra_gate'),
                                                               graph.number_of_edges('intra_module')))
        #filter out the irrelevant nodes that are not connected to any of the labeled PO
        remain_nodes,remove_nodes = graph_filter(graph)
        remain_nodes = remain_nodes.numpy().tolist()
        graph.remove_nodes(remove_nodes)
        is_module = filter_list(is_module,remain_nodes)
        nodes_type = filter_list(nodes_type,remain_nodes)
        nodes_name = filter_list(nodes_name, remain_nodes)

        nodes_list = th.tensor(range(graph.number_of_nodes()))
        PIs_nid = nodes_list[graph.ndata['is_pi']==1].numpy().tolist()
        PIs_name = [nodes_name[n] for n in PIs_nid]
        POs_nid = nodes_list[graph.ndata['is_po'] == 1].numpy().tolist()
        POs_name = [nodes_name[n] for n in POs_nid]

        nname2nid = {nm:nid for nid,nm in enumerate(nodes_name)}

        # get the topological level of the PO nodes
        topo = gen_topo(graph)
        nodes_level = th.zeros((len(nodes_name),1),dtype=th.float)
        PO2level = {}
        for l, nodes in enumerate(gen_topo(graph)):
            for n in nodes.numpy().tolist():
                nodes_level[n][0] = l
                if n in POs_nid:
                    PO2level[n] = l
        graph.ndata['level'] = nodes_level
        POs_level = [PO2level[n] for n in POs_nid]

        nodes_level = nodes_level.squeeze(1).numpy().tolist()
        graph.edges['intra_gate'].data['rating'] = edges_srcLevel_rating(graph,nodes_level,'intra_gate')
        graph.edges['intra_module'].data['rating'] = edges_srcLevel_rating(graph, nodes_level, 'intra_module')


        # filter out the POs that have abnormal label (large topo level but zero delay)
        remain_pos_idx = []
        for i,level in enumerate(POs_level):
            nid = POs_nid[i]
            PO_name = POs_name[i]
            PO_label = self.po_labels[PO_name]
            if (PO_label==0 and level>=2) or (PO_label==1 and level>=5) or (PO_label==2 and level>=10):
                print('\t removing PO:',PO_name,PO_label,level)
                graph.ndata['is_po'][nid] = 0
            else:
                remain_pos_idx.append(i)
        POs_level = filter_list(POs_level,remain_pos_idx)
        POs_name = filter_list(POs_name, remain_pos_idx)
        POs_nid = filter_list(POs_nid, remain_pos_idx)



        #print(graph.edges(etype='intra_module'))
        # src_m,dst_m = graph.edges(etype='intra_module')
        # src_m=src_m.numpy().tolist()
        # dst_m = dst_m.numpy().tolist()
        # visited = {}
        # for i,(src,dst) in enumerate(list(zip(src_m,dst_m))):
        #     if visited.get(dst,False):
        #         continue
        #     bit_pos = graph.edges['intra_module'].data['bit_position'][i].item()
        #     src_name = nodes_name[src]
        #     dst_name = nodes_name[dst]
        #
            # if bit_pos>32 and nodes_type[dst] in ['decoder','encoder']:
            #     visited[dst] = True
            #     print('+++++',bit_pos,src_name,dst_name)

                #exit()

        # save the necessary graph information
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
        graph_info['nicknames'] = self.nicknames

        print('\t post-filter: #node:{}, #edges:{}, {}'.format(graph.number_of_nodes(),graph.number_of_edges('intra_gate'),graph.number_of_edges('intra_module')))
        print('\t #PO:{}'.format(len(POs_name)))
        # print('\t',graph_info)

        return graph, graph_info

    def parse(self):
        #self.pi_delay,self.po_labels,_ = parse_golden(os.path.join(self.design_path,'golden_0.txt'))
        self.pi_delay, self.po_labels, _ = parse_golden(os.path.join(self.design_path, '{}_0'.format(self.design_name), 'golden.txt'))
        for p, d in self.pi_delay.items():
            assert d == 0, print("base case with non-zero input delay: {} {}".format(p, d))
        if self.pi_delay is None:
            return None,None
        graph, graph_info = self.parse_verilog()
        if graph is None:
            return None,None
        graph_info['base_po_labels'] = self.po_labels

        return graph,graph_info

def parse_golden(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    pi_delay = {}
    po_labels = {}
    po2delay2pis = {}
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

        words = line.split(' ')
        if len(words)==2:

            po,label = words
            po = po.replace(',', '')
            po_labels[po] = int(label)
        elif len(words)==3:
            po, pi, delay = words
            delay = int(delay)
            po = po.replace(',', '')
            pi = pi.replace(',', '')
            po_labels[po] = max(po_labels.get(po,0),delay)
            po2delay2pis[po] = po2delay2pis.get(po,{})
            po2delay2pis[po][delay] = po2delay2pis[po].get(delay,[])
            po2delay2pis[po][delay].append(pi)

        else:
            assert False, "wrong PO info: {}".format(line)

    if len(po2delay2pis)!=0:
        for po,label in po_labels.items():
            po_criticalPIs[po] = po2delay2pis[po][label]

    return pi_delay,po_labels,po_criticalPIs




def main():
    dataset = []
    num = 0

    #for subdir in ['']:
    for subdir in os.listdir(rawdata_path):
        subdir_path = os.path.join(rawdata_path,subdir)
        design2idx = {}
        for design in os.listdir(subdir_path):
            num += 1
            # if num not in list(range(300,401)):
            #    continue
            if int(design.split('_')[-1]) in [110,220,183,185,319,320,329,371,383,392,399]:
                continue
            # if int(design.split('_')[-1]) in [319,383,392]:
            #     continue
            # if '00099' not in design:
            #     continue

            design_dir = os.path.join(subdir_path,design)
            if not os.path.isdir(design_dir):
                continue
            print("-----Parsing {}-----".format(design))
            parser = Parser(design_dir)
            graph, graph_info = parser.parse()
            if graph is None or th.sum(graph.ndata['is_po']).item()==0:
                continue


            #label_files = [f for f in os.listdir(design_dir) if f.startswith('gold')]
            #case_indexs = [int(f.split('_')[-1].split('.')[0]) for f in label_files]

            label_files = [f for f in os.listdir(design_dir) if os.path.isdir(os.path.join(design_dir,f))]
            case_indexs = [int(f.split('_')[-1]) for f in label_files]
            case_indexs = sorted(case_indexs)

            graph_info['delay-label_pairs'] = []
            base_labels = {}
            for idx in case_indexs:
                # if idx==0:
                #     continue
                #golden_file_path = os.path.join(design_dir, 'golden_{}.txt'.format(idx))
                golden_file_path = os.path.join(design_dir, '{}_{}'.format(design,idx),'golden.txt')
                pi_delay,po_labels,po_criticalPIs = parse_golden(golden_file_path)
                #print(design_name,idx,po_labels)
                if len(po_labels)!=len(graph_info['base_po_labels']):
                    continue

                po_labels_residual = {p: d-graph_info['base_po_labels'][p] for (p,d) in po_labels.items()}
                PIs_delay, POs_label,POs_label_residual, POs = [], [],[], [ ]
                if pi_delay is None:
                    continue

                for node in graph_info["PIs_name"]:
                    PIs_delay.append(pi_delay[graph_info['nicknames'].get(node,node)])

                for node in graph_info['POs_name']:
                    POs_label.append(po_labels[node])
                    POs_label_residual.append(po_labels_residual[node])

                pi2po_edges = ([],[])
                for po, critical_pis in po_criticalPIs.items():
                    po_nid = graph_info['nname2nid'][po]
                    critical_pi_nids = [graph_info['nname2nid'][pi] for pi in critical_pis]
                    pi2po_edges[0].extend(critical_pi_nids)
                    pi2po_edges[1].extend([po_nid]*len(critical_pi_nids))

                if len(po_criticalPIs)==0:
                    nodes_delay = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    nodes_delay[graph.ndata['is_pi'] == 1] = th.tensor(PIs_delay, dtype=th.float).unsqueeze(1)
                    graph.ndata['delay'] = nodes_delay
                    pi2po_edges,edges_weight = get_pi2po_edges(graph,graph_info)


                #print(idx,len(PIs_delay),th.sum(graph.ndata['is_pi']).item(),len(POs_label),th.sum(graph.ndata['is_po']).item())
                assert len(PIs_delay) == th.sum(graph.ndata['is_pi']).item() and len(POs_label) == th.sum(graph.ndata['is_po']).item()
                graph_info['delay-label_pairs'].append((PIs_delay, POs_label,POs_label_residual,pi2po_edges))



            POs_base_label = []
            for node in graph_info['POs_name']:
                POs_base_label.append(graph_info['base_po_labels'][node])
            graph_info['base_po_labels'] = POs_base_label

            if len(graph_info['delay-label_pairs'])<=1:
                continue

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
    print(len(dataset))
    data_train = dataset[:int(0.7 * num_designs)]
    data_val = dataset[int(0.7 * num_designs):int(0.8 * num_designs)]
    data_test = dataset[int(0.8 * num_designs):]
    data_list = {
        'train': [g_info['design_name'] for g, g_info in data_train],
        'test': [g_info['design_name'] for g, g_info in data_test],
        'val': [g_info['design_name'] for g, g_info in data_val]}
    print('#train:{}, #val:{}, #test:{}'.format(len(data_train),len(data_val),len(data_test)))

    with open(os.path.join(data_savepath, 'split.pkl'), 'wb') as f:
        pickle.dump(data_list, f)
    with open(os.path.join(data_savepath,'data.pkl'),'wb') as f:
        pickle.dump(dataset,f)
    # with open(os.path.join(data_savepath,'graph.pkl'),'wb') as f:
    #     pickle.dump(final_dataset,f)

if __name__ == "__main__":
    stdout_f = os.path.join(data_savepath,'stdout.log')
    stderr_f = os.path.join(data_savepath, 'stderr.log')

    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
        main()
