from options import get_options
from model import *
import pickle
import numpy as np
import os
from random import shuffle
import random
import torch as th
from torch.utils.data import DataLoader,Dataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import softmax
import torch.nn as nn
import datetime
from torchmetrics import R2Score
import dgl
import tee



options = get_options()
device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")
R2_score = R2Score().to(device)
Loss = nn.MSELoss()

#ntype2id =  {'input': 0, "1'b0": 1, "1'b1": 2, 'add': 3, 'decoder': 4, 'and': 5, 'xor': 6, 'not': 7, 'or': 8, 'xnor': 9, 'encoder': 10, 'eq': 11, 'lt': 12, 'ne': 13, 'mux': 14}
with open(os.path.join(options.data_savepath, 'ntype2id.pkl'), 'rb') as f:
    ntype2id,ntype2id_gate,ntype2id_module = pickle.load(f)

print(ntype2id,ntype2id_gate,ntype2id_module)

def is_heter(graph):
    return len(graph._etypes)>1 or len(graph._ntypes)>1

def heter2homo(graph):
    src_module, dst_module = graph.edges(etype='intra_module', form='uv')
    src_gate, dst_gate = graph.edges(etype='intra_gate', form='uv')
    homo_g = dgl.graph((th.cat([src_module, src_gate]), th.cat([dst_module, dst_gate])))

    for key, data in graph.ndata.items():
        homo_g.ndata[key] = graph.ndata[key]

    return homo_g

def gen_topo(graph):
    if is_heter(graph):
        g = heter2homo(graph)
        topo = dgl.topological_nodes_generator(g)
    else:
        topo = dgl.topological_nodes_generator(graph)
    return topo


def load_data(usage):
    assert usage in ['train','val','test']
    print("----------------Loading {} data----------------".format(usage))
    data_path = options.data_savepath
    data_file = os.path.join(data_path, 'data_{}.pkl'.format(usage))
    # ntype_file = os.path.join(data_path, 'ntype2id.pkl')
    # with open(ntype_file, 'rb') as f:
    #     ntype2id = pickle.load(f)

    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    if options.flag_filter:
        with open(os.path.join(data_path,'delay_info_2.pkl'),'rb') as f:
            delay_info  = pickle.load(f)

        design_critical_nodes = {}
        for case, (_,POs_criticalpath,_,_) in delay_info.items():
            design_name,case_idx = case.split('_')
            design_critical_nodes[design_name] = design_critical_nodes.get(design_name,set())

            for path in POs_criticalpath:
                for nodes in path.values():
                    for n in nodes:
                        design_critical_nodes[design_name].add(n)

    loaded_data = []
    for  graph,graph_info in data:
        name2nid = {graph_info['nodes_name'][i]:i for i in range(len(graph_info['nodes_name']))}

        if options.flag_homo:
            graph = heter2homo(graph)

        graph.ndata['feat'] = graph.ndata['ntype']
        graph.ndata['feat_module'] = graph.ndata['ntype_module']
        graph.ndata['feat_gate'] = graph.ndata['ntype_gate']
        graph_info['POs_feat'] = graph_info['POs_level_max'].unsqueeze(-1)
        graph.ndata['h'] = th.ones((graph.number_of_nodes(), options.hidden_dim), dtype=th.float)
        # graph.ndata['is_po'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
        # graph.ndata['is_po'][graph_info['POs']] = 1
        #graph.ndata['label'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)

        #graph.ndata['label'][graph_info['POs']] = th.tensor(graph_info['delay-label_pairs'][0][1],dtype=th.float).unsqueeze(-1)
        graph.ndata['PO_feat'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
        graph.ndata['PO_feat'][graph.ndata['is_po']==1] = graph_info['POs_feat']


        # graph.ndata['delay'] = th.zeros((graph.number_of_nodes(),1),dtype=th.float)
        if len(graph_info['delay-label_pairs'][0][0])!= len(graph.ndata['is_pi'][graph.ndata['is_pi'] == 1]):
            print('skip',graph_info['design_name'])
            continue

        PIs = list(th.tensor(range(graph.number_of_nodes()))[graph.ndata['is_pi'] == 1])
        design_name = graph_info['design_name'].split('_')[-1]

        if options.flag_filter:
            critical_nodes = design_critical_nodes[design_name]
            critical_nodes = list(critical_nodes)
            critical_nodes = [name2nid[n] for n in critical_nodes]
            critical_nodes.extend(graph_info['POs'])
            PI_mask = []
            for j,pi in enumerate(PIs):
                if pi in critical_nodes:
                    PI_mask.append(j)
            non_critical_nodes = list(set(range(graph.number_of_nodes())) - set(critical_nodes))

            graph.remove_nodes(non_critical_nodes)
        else:
            PI_mask = list(range(len(PIs)))
        # print(graph.ndata['is_po'].numpy().tolist())
        # exit()
        #print(graph.number_of_nodes(), len(critical_nodes))

        #graph.ndata['delay'][graph.ndata['is_pi'] == 1] = th.tensor(graph_info['delay-label_pairs'][0][0], dtype=th.float).unsqueeze(-1)

        # ntype_onehot = th.zeros((graph.number_of_nodes(), len(ntype2id)), dtype=th.float)
        # for i, type in enumerate(graph_info['ntype']):
        #     ntype_onehot[i][ntype2id[type]] = 1
        # graph.ndata['feat'] = ntype_onehot
        graph_info['graph'] = graph
        graph_info['PI_mask'] = PI_mask
        loaded_data.append(graph_info)
        # if len(loaded_data)>5:
        #     break

    #loaded_data = loaded_data[:int(len(loaded_data)/10)]
    # batch_size = options.batch_size if usage=='train' else len(loaded_data)
    batch_size = options.batch_size
    drop_last = True if usage == 'train' else False
    #drop_last = False
    sampler = SubsetRandomSampler(th.arange(len(loaded_data)))
    idx_loader = DataLoader([i for i in range(len(loaded_data))], sampler=sampler, batch_size=batch_size,
                              drop_last=drop_last)
    return loaded_data,idx_loader

    # sampler = SubsetRandomSampler(th.arange(len(loaded_data)))
    # # batch_size = options.batch_size if usage=='train' else len(loaded_data)
    # batch_size = options.batch_size
    # drop_last = True if usage=='train' else False
    # loader = DataLoader(MyLoader(loaded_data), sampler=sampler, batch_size=batch_size,
    #                          drop_last=drop_last)

    # return loader



def init_model(options):
    model = TimeConv(
            infeat_dim1=len(ntype2id_gate),
            infeat_dim2=len(ntype2id_module),
            hidden_dim=options.hidden_dim,
            flag_splitfeat=options.split_feat,
            attn_choice=options.attn_choice,
            flag_homo=options.flag_homo,
            flag_global=options.flag_global,
            flag_attn=options.flag_attn
        ).to(device)
    # model_base = TimeConv(
    #     infeat_dim=len(ntype2id),
    #     hidden_dim=options.hidden_dim
    # ).to(device)
    print("creating model:")
    print(model)

    return model


def init(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test(model,test_data,test_idx_loader):

    with (th.no_grad()):
        total_num, total_loss, total_r2 = 0, 0.0, 0
        labels,labels_hat = None,None
        for batch, idxs in enumerate(test_idx_loader):
            sampled_data = []
            idxs = idxs.numpy().tolist()
            num_cases = 100
            for idx in idxs:
                data = test_data[idx]
                num_cases = min(num_cases,len(data['delay-label_pairs']))
                sampled_data.append(test_data[idx])
            #print('test num_cases', num_cases)
            if options.target_base: num_cases = 1
            for i in range(num_cases):
                new_sampled_data = []
                graphs = []
                for data in sampled_data:
    
                    PIs_delay, POs_label, _ = data['delay-label_pairs'][i]
                    if options.target_base:
                        POs_label = data['base_po_labels']


                    graph = data['graph']
                    graph.ndata['label'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    graph.ndata['label'][graph.ndata['is_po'] == 1] = th.tensor(POs_label, dtype=th.float).unsqueeze(-1)
                    graph.ndata['delay'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    graph.ndata['delay'][graph.ndata['is_pi'] == 1] = th.tensor(PIs_delay, dtype=th.float)[data['PI_mask']].unsqueeze(-1)

                    data['graph'] = graph
                    graphs.append(graph)
                sampled_graphs = dgl.batch(graphs)
                sampled_graphs = sampled_graphs.to(device)
                graphs_info = {}

                topo_levels = gen_topo(sampled_graphs)

                graphs_info['is_heter'] = is_heter(sampled_graphs)
                graphs_info['topo'] = [l.to(device) for l in topo_levels]
                # graphs_info['POs_feat'] = POs_feat.to(device)
                graphs_info['POs'] = (sampled_graphs.ndata['is_po']==1).squeeze(-1).to(device)
                POs_topolevel = sampled_graphs.ndata['PO_feat'][sampled_graphs.ndata['is_po'] == 1].to(device)

                #graphs_info['POs_feat'] = th.cat((POs_topolevel,POs_propdelay),dim=1)
                graphs_info['POs_feat'] = POs_topolevel

                cur_labels_hat = model(sampled_graphs, graphs_info)
                cur_labels = sampled_graphs.ndata['label'][graphs_info['POs']].to(device)
                if labels_hat is None:
                    labels_hat = cur_labels_hat
                    labels = cur_labels
                else:
                    labels_hat = th.cat((labels_hat, cur_labels_hat), dim=0)
                    labels = th.cat((labels, cur_labels), dim=0)

        test_loss = Loss(labels_hat, labels).item()
        test_r2 = R2_score(labels_hat, labels).item()
        test_mape = th.mean(th.abs(labels_hat[labels != 0] - labels[labels != 0]) / labels[labels != 0])
        ratio = labels_hat[labels != 0] / labels[labels != 0]
        min_ratio = th.min(ratio)
        max_ratio = th.max(ratio)
        return test_loss, test_r2,test_mape,min_ratio,max_ratio

def train(model):
    print(options)
    th.multiprocessing.set_sharing_strategy('file_system')

    train_data,train_idx_loader = load_data('train')
    val_data,val_idx_loader = load_data('val')
    test_data,test_idx_loader = load_data('test')
    # train_loader = DataLoader(list(range(len(data_train))), batch_size=options.batch_size, shuffle=True, drop_last=False)
    print("Data successfully loaded")

    # set the optimizer
    optim = th.optim.Adam(
        model.parameters(), options.learning_rate, weight_decay=options.weight_decay
    )
    model.train()


    print("----------------Start training----------------")

    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    num_traindata = len(train_data)
    for epoch in range(options.num_epoch):
        print('Epoch {} ------------------------------------------------------------'.format(epoch+1))
        total_num,total_loss, total_r2 = 0,0.0,0

        for batch, idxs in enumerate(train_idx_loader):
            sampled_data = []

            idxs = idxs.numpy().tolist()
            num_cases = 100
            for idx in idxs:
                data = train_data[idx]
                num_cases = min(num_cases,len(data['delay-label_pairs']))
                shuffle(train_data[idx]['delay-label_pairs'])
                sampled_data.append(train_data[idx])

            if options.target_base: num_cases = 1

            #print('num_cases',num_cases)
            for i in range(num_cases):

                new_sampled_data = []
                graphs = []
                for data in sampled_data:

                    PIs_delay, POs_label, _ = data['delay-label_pairs'][i]
                    if options.target_base:
                        POs_label =  data['base_po_labels']

                    # for j in range(len(POs_label)):
                    #     POs_label[j] += data['base_po_labels'][j]

                    graph = data['graph']
                    graph.ndata['label'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    graph.ndata['label'][graph.ndata['is_po'] == 1] = th.tensor( POs_label,dtype=th.float).unsqueeze(-1)
                    graph.ndata['delay'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    graph.ndata['delay'][graph.ndata['is_pi'] == 1] = th.tensor(PIs_delay,dtype=th.float)[data['PI_mask']].unsqueeze(-1)
                    #print(POs_prop_delay)

                    data['graph'] = graph
                    graphs.append(graph)
                sampled_graphs = dgl.batch(graphs)


                sampled_graphs = sampled_graphs.to(device)
                graphs_info = {}
                topo_levels = gen_topo(sampled_graphs)

                graphs_info['is_heter'] = is_heter(sampled_graphs)
                graphs_info['topo'] = [l.to(device) for l in topo_levels]
                # graphs_info['POs_feat'] = POs_feat.to(device)
                graphs_info['POs'] = (sampled_graphs.ndata['is_po']==1).squeeze(-1).to(device)
                POs_topolevel = sampled_graphs.ndata['PO_feat'][sampled_graphs.ndata['is_po'] == 1].to(device)

                #print(len(graphs_info['POs']),len(sampled_graphs.ndata['PO_feat'][sampled_graphs.ndata['is_po'] == 1]))
                #graphs_info['POs_feat'] = th.cat((POs_topolevel, POs_propdelay), dim=1)
                graphs_info['POs_feat'] = POs_topolevel
                #graphs_info['POs_feat'] = sampled_graphs.ndata['PO_feat'][graphs_info['POs']].to(device)
                labels_hat = model(sampled_graphs, graphs_info)
                labels = sampled_graphs.ndata['label'][graphs_info['POs']].to(device)



                total_num += len(labels)
                train_loss = Loss(labels_hat, labels)

                train_r2 = R2_score(labels_hat, labels).to(device)
                train_mape = th.mean(th.abs(labels_hat[labels!=0]-labels[labels!=0])/labels[labels!=0])
                ratio = labels_hat[labels!=0] / labels[labels!=0]
                min_ratio = th.min(ratio)
                max_ratio = th.max(ratio)
                if i==num_cases-1: print('{}/{} train_loss:{:.3f}\ttrain_r2:{:.3f}\ttrain_mape:{:.3f}, ratio:{:.2f}-{:.2f}'.format((batch+1)*options.batch_size,num_traindata,train_loss.item(),train_r2.item(),train_mape.item(),min_ratio,max_ratio))
                optim.zero_grad()
                train_loss.backward()
                optim.step()

        val_loss, val_r2,val_mape,val_min_ratio,val_max_ratio = test(model, val_data,val_idx_loader)
        test_loss, test_r2,test_mape,test_min_ratio,test_max_ratio = test(model,test_data,test_idx_loader)
        print('End of epoch {}'.format(epoch))
        print('\tval:  loss={:.3f}\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(val_loss,val_r2,val_mape,val_min_ratio,val_max_ratio))
        print('\ttest: loss={:.3f}\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(test_loss,test_r2,test_mape,test_min_ratio,test_max_ratio))
        if options.checkpoint:
            save_path = '../checkpoints/{}'.format(options.checkpoint)
            th.save(model.state_dict(), os.path.join(save_path,"{}.pth".format(epoch)))
            with open(os.path.join(checkpoint_path,'res.txt'),'a') as f:
                f.write('Epoch {}, val: {:.3f},{:.3f}; test:{:.3f},{:.3f}\n'.format(epoch,val_r2,val_mape,test_r2,test_mape))


if __name__ == "__main__":
    seed = random.randint(1, 10000)
    init(seed)
    if options.test_iter:
        assert options.checkpoint, 'no checkpoint dir specified'
        model_save_path = '../checkpoints/{}/{}.pth'.format(options.checkpoint, options.test_iter)
        assert os.path.exists(model_save_path), 'start_point {} of checkpoint {} does not exist'.\
            format(options.test_iter, options.checkpoint)
        input_options = options
        options = th.load('../checkpoints/{}/options.pkl'.format(options.checkpoint))
        options.data_savepath = input_options.data_savepath
        options.target_base = input_options.target_base
        options.flag_filter = input_options.flag_filter

        model = init_model(options)
        model = model.to(device)
        model.load_state_dict(th.load(model_save_path))
        test_data,test_idx_loader = load_data('test')
        test_loss, test_r2,test_mape,test_min_ratio,test_max_ratio = test(model, test_data,test_idx_loader)
        print(
            '\ttest: loss={:.3f}\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(test_loss, test_r2,
                                                                                                     test_mape,
                                                                                                     test_min_ratio,
                                                                                                     test_max_ratio))




    elif options.checkpoint:
        print('saving logs and models to ../checkpoints/{}'.format(options.checkpoint))
        checkpoint_path = '../checkpoints/{}'.format(options.checkpoint)
        stdout_f = '../checkpoints/{}/stdout.log'.format(options.checkpoint)
        stderr_f = '../checkpoints/{}/stderr.log'.format(options.checkpoint)
        os.makedirs(checkpoint_path)  # exist not ok
        th.save(options, os.path.join(checkpoint_path, 'options.pkl'))
        with open(os.path.join(checkpoint_path,'res.txt'),'w') as f:
            pass
        with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
            model = init_model(options)
            model = model.to(device)
            print('seed:', seed)
            train(model)

    else:
        print('No checkpoint is specified. abandoning all model checkpoints and logs')
        model = init_model(options)
        model = model.to(device)
        train(model)