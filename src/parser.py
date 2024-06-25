import torch as th
import dgl
import os
import pickle
from options import get_options
from random import shuffle

options = get_options()
rawdata_path = options.rawdata_path
data_savepath = options.data_savepath
os.makedirs(data_savepath,exist_ok=True)

ntype_file = os.path.join(data_savepath,'ntype2id.pkl')
if os.path.exists(ntype_file):
    with open(ntype_file,'rb') as f:
        ntype2id = pickle.load(f)
else:
    ntype2id = {
        'input':0,
        "1'b0":1,
        "1'b1":2,
    }


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
                    self.nodes[node] = {'ntype': "1'b0", 'is_po': False}
                    res.append(node)
                elif v == "1":
                    self.const1_index += 1
                    node = "1'b1_{}".format(self.const1_index)
                    self.nodes[node] = {'ntype': "1'b1", 'is_po': False}
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
                    self.nodes[node_name] = {'ntype':wire_type,'is_po':wire_type=='output'}
                    self.wires_width[node_name] = (0,0)
                # e.g., wire [4:0] do_0_b;
                else:
                    sentence = sentence.strip()
                    wire_type, bit_range, wire_name = sentence.split(' ');
                    high_bit,low_bit = bit_range[bit_range.find('[')+1:bit_range.rfind(']')].split(':')
                    wire_type = wire_type.replace(' ','')
                    wire_name = wire_name.replace(';','')
                    self.wires_width[wire_name] = (int(low_bit),int(high_bit))
                    for i in range(int(low_bit),int(high_bit)+1):
                        node_name = '{}[{}]'.format(wire_name,i)
                        self.nodes[node_name] = {'ntype': wire_type,'is_po':wire_type=='output'}


            else:
                fo2fi = {}  # {fanout_name:fanin_list}
                sentence = sentence.replace('\n','').strip()
                # print(sentence)
                # get the gate type
                gate = sentence[:sentence.find('(')]
                gate_type, gate_name = gate.strip().split(' ')

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
                        self.nodes[n]['ntype'] = gate_type
                        ntype2id[gate_type] = ntype2id.get(gate_type,len(ntype2id))
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
                # deal with add, less than, not equal gate
                elif 'add' in gate_type.lower() or 'lt' in gate_type.lower() or 'ne' in gate_type.lower():
                    # get the io wires list
                    io_wires = sentence[sentence.find('(') + 1:].strip()
                    io_wires = io_wires.split('),')
                    io_wires = [p.replace(' ', '') for p in io_wires]

                    io_wires = {p[1:p.find('(')]: p[p.find('(') + 1:].strip().replace(')', '') for p in io_wires}
                    io_nodes = {p: self.parse_wire(w) for p, w in io_wires.items()}
                    # get the output nodes, and set their gate type;
                    fanout_nodes = io_nodes['o']
                    for n in fanout_nodes:
                        self.nodes[n]['ntype'] = gate_type
                        ntype2id[gate_type] = ntype2id.get(gate_type, len(ntype2id))
                        fo2fi[n] = []
                    # add the edges between fanin nodes and fanout nodes
                    for port, fanin_nodes in io_nodes.items():
                        if port == 'o':
                            continue
                        # each output bit is related to all input bits up to its own bit width.
                        for i, fanout_node in enumerate(fanout_nodes):
                            related_fanin_nodes = []
                            for fanin_node in fanin_nodes:
                                fanin_bit_index = int(
                                    fanin_node.split('[')[-1].split(']')[0]) if '[' in fanin_node else 0
                                if fanin_bit_index <= i:
                                    related_fanin_nodes.append(fanin_node)
                            fo2fi[fanout_node].extend(related_fanin_nodes)
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
                    else:
                        ntype2id[gate_type] = ntype2id.get(gate_type, len(ntype2id))
                        fo2fi[fanout_node] = fanin_nodes

                # add the edges from fanins to fanouts
                for fanout, fanins in fo2fi.items():
                    for fanin in fanins:
                        self.edges.append(
                            (fanin,fanout,{})
                        )
                        # print("\te: {} -> {}".format(fanin, fanout))




        self.nodes = {n : self.nodes[n] for n in self.nodes.keys() if self.nodes[n]['ntype'] not in ['wire',None]}

        new_edges = []
        for src,dst, e_info in self.edges:
            if buf_o2i.get(src,None) is not None:
                new_edges.append((buf_o2i[src],dst,e_info))
            else:
                new_edges.append((src,dst,e_info))
        self.edges = new_edges


        # print(self.nodes)
        # for e in self.edges:
        #     print(e)

        # construct the graph
        src_nodes, dst_nodes = [],[]
        graph_info = {}
        node2nid = {}
        nid2node = {}
        nodes_type,nodes_delay,nodes_name,POs, POs_label = [],[],[],[],[]
        POs_name = []
        is_po = []
        # print(self.po_labels)
        for node,node_info in self.nodes.items():
            nid = len(node2nid)
            node2nid[node] = nid
            nid2node[node2nid[node]] = node
            nodes_name.append(node)
            nodes_type.append(node_info['ntype'])
            # set the PI delay
            nodes_delay.append(self.pi_delay.get(node,0))
            # set the PO label
            if node_info['is_po']:
                is_po.append(1)
                nickname = node_info.get('nickname',None)
                # print(node,nickname)
                if nickname is not None:
                    node = nickname
                if self.po_labels.get(node,None) is not None:
                    POs_name.append(node)
                    POs.append(nid)
                    POs_label.append(self.po_labels[node])
            else:
                is_po.append(0)
        # print({nodes_name[i]:nodes_delay[i] for i in range(len(nodes_name))})

        # get the src_node list and dst_node lsit
        for eid, (src, dst, edict) in enumerate(self.edges):
            src_nodes.append(node2nid[src])
            dst_nodes.append(node2nid[dst])

        # print(self.nodes)
        graph = dgl.graph(
            (src_nodes, dst_nodes), num_nodes=len(node2nid)
        )
        topo = dgl.topological_nodes_generator(graph)
        PO2level = {}
        j = 0
        for l,nodes in enumerate(topo):
            for n in nodes.numpy().tolist():
                if n in POs:
                    PO2level[n] = l
        POs_level = [PO2level[n] for n in POs]
        graph_info['topo'] = topo
        graph_info['ntype'] = nodes_type
        graph_info['nname'] = nodes_name
        graph_info['POs'] = POs
        graph_info['POs_label'] = th.tensor(POs_label,dtype=th.float)
        graph_info['POs_level_max'] = th.tensor(POs_level,dtype=th.float)
        graph_info['case_name'] = self.case_name
        graph_info['design_name'] = self.design_name
        graph.ndata['is_po'] = th.tensor(is_po)
        graph.ndata['delay'] = th.tensor(nodes_delay,dtype=th.float).reshape((len(nodes_delay),1))

        # print(POs)
        # print(topo)
        POs_level_min = []     # record the shortest path length from an PI to any PI
        for po in POs:
            l = 0
            # print(po)
            predecessors = graph.predecessors(po).numpy().tolist()
            # predecessors_name = [nid2node[n] for n in predecessors]
            while len(predecessors) !=0:
                l +=1
                new_predecessors = []
                for p in predecessors:
                    pre_preds = graph.predecessors(p).numpy().tolist()
                    if len(pre_preds)==0:
                        break
                    new_predecessors.extend(pre_preds)
                # predecessors_name = [nid2node[n] for n in new_predecessors]
                predecessors = new_predecessors
                # print('\t', predecessors_name)
            POs_level_min.append(l)
        graph_info['POs_level_min'] = th.tensor(POs_level_min,dtype=th.float)

        # print('\t',graph)
        print('\t #node:{}, #edges:{}'.format(graph.number_of_nodes(),graph.number_of_edges()))
        # print('\t',graph_info)
        print('\t POs: levels={}, labels={}'.format(POs_level,POs_label))
        print('\t',POs_level_min)
        return graph, graph_info

    def parse_golden(self):
        file_path = os.path.join(self.design_path,'golden.txt')
        with open(file_path, 'r') as f:
            content = f.read()

        pi_content,po_content = content.split('// pin to pin level synthesised\n')

        for line in pi_content.split('\n'):
            if '//' in line or len(line)==0:
                continue
            pi,delay = line.split(' ')
            self.pi_delay[pi] = int(delay)
        for line in po_content.split('\n'):
            if '//' in line or len(line)==0:
                continue

            po, label = line.split(' ')
            po = po.replace(',','')
            self.po_labels[po] = int(label)
        # print(self.pi_delay)
        # print(self.po_labels)

    def parse(self):
        self.parse_golden()
        graph, graph_info = self.parse_verilog()

        return graph, graph_info



def main():
    dataset = []
    num = 0
    for subdir in os.listdir(rawdata_path):
        subdir_path = os.path.join(rawdata_path,subdir)
        for design in os.listdir(subdir_path):
            # design_name,index = design.split('_')[-2:]
            # print(design_name,index)
            # exit()
            # if '190' not in design: continue
            print("Processing {}/{}".format(subdir,design))
            design_dir = os.path.join(subdir_path,design)
            if not os.path.isdir(design_dir):
                continue
            parser = Parser(subdir,design_dir)
            graph, graph_info = parser.parse()

            dataset.append((graph,graph_info))

        # exit()
    if not os.path.exists(ntype_file):
        with open(ntype_file,'wb') as f:
            pickle.dump(ntype2id,f)
    print('ntypes:',ntype2id)

    final_dataset = {}
    for graph, graph_info in dataset:
        ntype_onehot = th.zeros((graph.number_of_nodes(), len(ntype2id)), dtype=th.float)
        for i, type in enumerate(graph_info['ntype']):
            ntype_onehot[i][ntype2id[type]] = 1
        graph.ndata['feat'] = ntype_onehot
        design_name = graph_info['design_name']
        final_dataset[design_name] = final_dataset.get(design_name,[])
        final_dataset[design_name].append((graph,graph_info))
        # print(design_name,len(final_dataset[design_name]))
    final_dataset = list(final_dataset.items())
    shuffle(final_dataset)
    num_samples = len(final_dataset)
    # print(num_samples)
    # print(len(final_dataset[0][1]))
    split_ratio = [0.7,0.1,0.2]
    data_train,data_val,data_test = [],[],[]
    for i in range(0,int(num_samples*split_ratio[0])):
        data_train.extend(final_dataset[i][1])
    for i in range(int(num_samples*split_ratio[0]), int(num_samples*(split_ratio[0]+split_ratio[1]))):
        data_val.extend(final_dataset[i][1])
    for i in range(int(num_samples * (split_ratio[0] + split_ratio[1])),num_samples):
        data_test.extend(final_dataset[i][1])

    print('#train:{}, #val:{}, #test:{}'.format(len(data_train),len(data_val),len(data_test)))
    with open(os.path.join(data_savepath,'data_train.pkl'),'wb') as f:
        pickle.dump(data_train,f)
    with open(os.path.join(data_savepath,'data_val.pkl'),'wb') as f:
        pickle.dump(data_val,f)
    with open(os.path.join(data_savepath,'data_test.pkl'),'wb') as f:
        pickle.dump(data_test,f)
    with open(os.path.join(data_savepath,'data.pkl'),'wb') as f:
        pickle.dump(final_dataset,f)
    # with open(os.path.join(data_savepath,'graph.pkl'),'wb') as f:
    #     pickle.dump(final_dataset,f)

if __name__ == "__main__":
    main()
