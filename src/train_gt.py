import matplotlib.pyplot as plt
import torch
import dgl
print(dgl.__version__)
print(torch.__version__)

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
from torch.utils.data.sampler import SubsetRandomSampler,Sampler
from torch.nn.functional import softmax
import torch.nn as nn
import datetime
from torchmetrics import R2Score
import dgl
import tee
from utils import *
import itertools

from gtmodel import GTModel
from dgl import LapPE
from dgl.nn import LapPosEncoder



options = get_options()
device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")
#device = th.device("cpu")
R2_score = R2Score().to(device)
Loss = nn.MSELoss()
Loss = nn.L1Loss()


with open(os.path.join(options.data_savepath, 'ntype2id.pkl'), 'rb') as f:
    ntype2id,ntype2id_gate,ntype2id_module = pickle.load(f)
num_gate_types = len(ntype2id_gate)
num_gate_types -= 3
num_module_types = len(ntype2id_module)
# print(num_gate_types,num_module_types)
# # print(ntype2id)
# print(ntype2id,ntype2id_gate,ntype2id_module)
# exit()

data_path = options.data_savepath
if data_path.endswith('/'):
    data_path = data_path[:-1]
data_file = os.path.join(data_path, 'data.pkl')
#split_file = os.path.join(data_path, 'split.pkl')
split_file = os.path.join(os.path.split(data_path)[0], 'split_new.pkl')

with open(data_file, 'rb') as f:
    data_all = pickle.load(f)
    design_names = [d[1]['design_name'].split('_')[-1] for d in data_all]

with open(split_file, 'rb') as f:
    split_list = pickle.load(f)


def cat_tensor(t1,t2):
    if t1 is None:
        return t2
    elif t2 is None:
        return t1
    else:
        return th.cat((t1,t2),dim=0)
# print(split_list)
# exit()
def load_data(usage,flag_quick=True,flag_inference=False):

    assert usage in ['train','val','test']

    target_list = split_list[usage]
    target_list = [n.split('_')[-1] for n in target_list]

    data = [d for i,d in enumerate(data_all) if design_names[i] in target_list]
    case_range = (0, 100)
    if flag_quick:
        if usage == 'train':
            case_range = (0,20)
        else:
            case_range = (0, 40)
    print("------------Loading {}_data #{} {}-------------".format(usage,len(data),case_range))

    loaded_data = []
    for  graph,graph_info in data:

        graph = heter2homo(graph)

        if options.inv_choice!=-1:
            graph.edges['intra_module'].data['is_inv'] = graph.edges['intra_module'].data['is_inv'].unsqueeze(1)
            graph.edges['intra_gate'].data['is_inv'] = graph.edges['intra_gate'].data['is_inv'].unsqueeze(1)
        #graph.ndata['feat'] = graph.ndata['ntype']
        graph.ndata['feat_type'] = graph.ndata['ntype'][:,3:]

        if len(graph_info['delay-label_pairs'][0][0])!= len(graph.ndata['is_pi'][graph.ndata['is_pi'] == 1]):
            print('skip',graph_info['design_name'])
            continue

        graph_info['graph'] = graph
        graph_info['delay-label_pairs'] = graph_info['delay-label_pairs'][case_range[0]:case_range[1]]
        loaded_data.append(graph_info)

    batch_size = options.batch_size
    if not flag_inference and (not options.flag_reverse or options.flag_path_supervise) and usage!='train':
        batch_size = len(loaded_data)


    return loaded_data

def get_idx_loader(data,batch_size):
    drop_last = False
    sampler = SubsetRandomSampler(th.arange(len(data)))
    idx_loader = DataLoader([i for i in range(len(data))], sampler=sampler, batch_size=batch_size,
                            drop_last=drop_last)
    return idx_loader

def collate(graphs):
    # compute shortest path features, can be done in advance
    for g in graphs:
        spd = dgl.shortest_dist(g, root=None, return_paths=False)
        g.ndata["spd"] = spd
        #print(spd,spd.shape)

    num_graphs = len(graphs)
    num_nodes = [g.num_nodes() for g in graphs]
    max_num_nodes = max(num_nodes)

    POs_mask = []
    attn_mask = th.zeros(num_graphs, max_num_nodes, max_num_nodes)
    node_feat = []
    in_degree, out_degree = [], []
    path_data = []
    # Since shortest_dist returns -1 for unreachable node pairs and padded
    # nodes are unreachable to others, distance relevant to padded nodes
    # use -1 padding as well.
    dist = -th.ones(
        (num_graphs, max_num_nodes, max_num_nodes), dtype=th.long
    )

    for i in range(num_graphs):
        # A binary mask where invalid positions are indicated by True.
        # Avoid the case where all positions are invalid.
        attn_mask[i, :, num_nodes[i] + 1 :] = 1
        nodes_list = th.tensor(range(graphs[i].number_of_nodes()))
        POs = nodes_list[graphs[i].ndata["is_po"]==1]
        POs_mask.append(POs)

        # +1 to distinguish padded non-existing nodes from real nodes
        node_feat.append(graphs[i].ndata["feat"] + 1)

        # 0 for padding
        in_degree.append(
            th.clamp(graphs[i].in_degrees() + 1, min=0, max=512)
        )
        out_degree.append(
            th.clamp(graphs[i].out_degrees() + 1, min=0, max=512)
        )

        dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i].ndata["spd"]

    # node feat padding
    node_feat = th.nn.utils.rnn.pad_sequence(node_feat, batch_first=True)
    #POs_mask = th.nn.utils.rnn.pad_sequence(POs_mask, batch_first=True)

    # degree padding
    in_degree = th.nn.utils.rnn.pad_sequence(in_degree, batch_first=True)
    out_degree = th.nn.utils.rnn.pad_sequence(out_degree, batch_first=True)

    # return {
    #     'POs_mask': POs_mask,
    #     'node_feat': node_feat,
    #     'in_degree': in_degree,
    #     'out_degree': out_degree,
    #     'attn_mask': attn_mask,
    #     'dist': dist
    # }

    return {
        'POs_mask':POs_mask,
        'node_feat':node_feat.to(device),
        'in_degree':in_degree.to(device),
        'out_degree':out_degree.to(device),
        'attn_mask':attn_mask.to(device),
        'dist':dist.to(device)
    }

def init_model(options):
    # model = Graphormer(infeat_dim=num_gate_types+num_module_types+1,
    #                    feat_dim=128,
    #                    hidden_dim=256)
    model = GTModel(in_size=num_gate_types+num_module_types+1,out_size=1,hidden_size=256)
    print("creating model:")
    print(model)

    return model


def init(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_pos_encoding(g):
    transform = LapPE(k=5, feat_name='eigvec', eigval_name='eigval', padding=True)
    g = dgl.graph(([0, 1, 2, 3, 4, 2, 3, 1, 4, 0], [2, 3, 1, 4, 0, 0, 1, 2, 3, 4]))
    g = transform(g)
    eigvals, eigvecs = g.ndata['eigval'], g.ndata['eigvec']
    transformer_encoder = LapPosEncoder(
        model_type="Transformer", num_layer=3, k=5, dim=16, n_head=4
    )
    pos_encoding=transformer_encoder(eigvals, eigvecs)
    # deepset_encoder = LapPosEncoder(
    #     model_type="DeepSet", num_layer=3, k=5, dim=16, num_post_layer=2
    # )
    # pos_encoding=deepset_encoder(eigvals, eigvecs)

    return pos_encoding


def gather_data(sampled_data,sampled_graphs,idx):

    POs_label_all, PIs_delay_all = None, None
    for data in sampled_data:
        PIs_delay, POs_label, POs_baselabel, pi2po_edges = data['delay-label_pairs'][idx][:4]

        POs_label_all = cat_tensor(POs_label_all, th.tensor(POs_label, dtype=th.float).unsqueeze(-1).to(device))
        PIs_delay_all = cat_tensor(PIs_delay_all, th.tensor(PIs_delay, dtype=th.float).unsqueeze(-1).to(device))


    #graphs_info['label'] = POs_label_all
    feat_delay = th.zeros((sampled_graphs.number_of_nodes(), 1), dtype=th.float).to(device)
    feat_delay[sampled_graphs.ndata['is_pi'] == 1] = PIs_delay_all
    sampled_graphs.ndata['feat'] = th.cat((sampled_graphs.ndata['feat_type'],feat_delay),dim=1)


    return POs_label_all, sampled_graphs




def test(model,test_data,batch_size):


    test_idx_loader = get_idx_loader(test_data, batch_size)
    model.flag_train = False
    with (th.no_grad()):
        labels_all,labels_hat_all = None,None
        # for i in range(0,len(test_data),batch_size):
        #     idxs = list(range(i,min(i+batch_size,len(test_data))))
        for batch, idxs in enumerate(test_idx_loader):
            idxs = idxs.numpy().tolist()
            sampled_data = []
            num_cases = 100
            graphs = []
            for idx in idxs:
                data = test_data[idx]
                num_cases = min(num_cases,len(data['delay-label_pairs']))
                sampled_data.append(test_data[idx])
                graphs.append(data['graph'])
            graphs_info = collate(graphs)




            for i in range(num_cases):
                node_delay = []
                labels = None
                torch.cuda.empty_cache()
                for data in sampled_data:
                    g = data['graph']
                    PIs_delay, POs_label, _, _ = data['delay-label_pairs'][i][:4]
                    PIs_delay = th.tensor(PIs_delay, dtype=th.float).unsqueeze(-1).to(device)
                    POs_label = th.tensor(POs_label, dtype=th.float).unsqueeze(-1).to(device)
                    labels = cat_tensor(labels, POs_label)
                    delay = th.zeros((g.number_of_nodes(), 1), dtype=th.float).to(device)
                    delay[g.ndata['is_pi'] == 1] = PIs_delay
                    node_delay.append(delay + 1)
                node_delay_t = th.nn.utils.rnn.pad_sequence(node_delay, batch_first=True)
                graphs_info['ndoe_feat'] = th.cat((graphs_info['node_feat'], node_delay_t), dim=2)

                labels_hat = model(graphs_info)

                labels_hat_all = cat_tensor(labels_hat_all,labels_hat)
                labels_all = cat_tensor(labels_all,labels)

        test_loss = Loss(labels_hat_all, labels_all).item()
        test_r2 = R2_score(labels_hat_all, labels_all).item()
        test_mape = th.mean(th.abs(labels_hat_all[labels_all != 0] - labels_all[labels_all != 0]) / labels_all[labels_all != 0])
        ratio = labels_hat_all[labels_all != 0] / labels_all[labels_all != 0]
        min_ratio = th.min(ratio)
        max_ratio = th.max(ratio)

        return test_loss, test_r2,test_mape,min_ratio,max_ratio


def train(model):
    print(options)
    th.multiprocessing.set_sharing_strategy('file_system')

    train_data = load_data('train',options.quick)
    val_data = load_data('val',options.quick)
    test_data = load_data('test',options.quick)
    print("Data successfully loaded")

    train_idx_loader = get_idx_loader(train_data,options.batch_size)

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
            torch.cuda.empty_cache()
            sampled_data = []

            idxs = idxs.numpy().tolist()
            num_cases = 1000
            graphs = []

            for idx in idxs:
                data = train_data[idx]
                num_cases = min(num_cases,len(data['delay-label_pairs']))
                #shuffle(train_data[idx]['delay-label_pairs'])
                sampled_data.append(train_data[idx])

                data['graph'].ndata['PE'] = get_pos_encoding(data['graph'])
                graphs.append(data['graph'])

            sampled_graphs = dgl.batch(graphs)
            sampled_graphs = sampled_graphs.to(device)



            graphs_info = collate(graphs)


            total_labels,total_labels_hat = None,None

            for i in range(num_cases):
                labels = None
                node_delay = []
                torch.cuda.empty_cache()
                POs_label_all, sampled_graphs = gather_data(sampled_data, sampled_graphs, i)

                labels_hat = model(sampled_graphs,sampled_graphs.ndata['feat'],sampled_graphs.ndata['PE'])

                train_loss = Loss(labels_hat, labels)
                print(train_loss)

                total_labels = cat_tensor(total_labels, labels)
                total_labels_hat = cat_tensor(total_labels_hat, labels_hat)


                if i==num_cases-1:
                    train_r2 = R2_score(total_labels_hat, total_labels)
                    train_mape = th.mean(th.abs(total_labels_hat[total_labels != 0] - total_labels[total_labels != 0]) / total_labels[total_labels != 0])
                    ratio = total_labels_hat[total_labels != 0] / total_labels[total_labels != 0]
                    min_ratio = th.min(ratio)
                    max_ratio = th.max(ratio)
                    #print(data['design_name'],len(total_labels),num_POs)
                    #print(model.attention_vector_g[0].item(),model.attention_vector_m[0].item())
                    print('{}/{} train_loss:{:.3f}\ttrain_r2:{:.3f}\ttrain_mape:{:.3f}, ratio:{:.2f}-{:.2f}'.format((batch+1)*options.batch_size,num_traindata,train_loss.item(),train_r2.item(),train_mape.item(),min_ratio,max_ratio))

                if len(labels_hat) ==0:
                    continue

                optim.zero_grad()
                train_loss.backward()
                optim.step()
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        val_loss, val_r2,val_mape,val_min_ratio,val_max_ratio = test(model, val_data,options.batch_size)
        test_loss, test_r2,test_mape,test_min_ratio,test_max_ratio = test(model,test_data,options.batch_size)
        torch.cuda.empty_cache()
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

        options.batch_size = input_options.batch_size
        options.gpu = input_options.gpu
        options.quick = input_options.quick

        print(options)
        # exit()
        model = init_model(options)

        model = model.to(device)
        model.load_state_dict(th.load(model_save_path,map_location='cuda:{}'.format(options.gpu)))
        usages = ['train','test','val']
        usages = ['test']
        for usage in usages:
            flag_save = True
            save_file_dir = options.checkpoint
            test_data = load_data(usage,options.quick,flag_inference)

            #test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = test(model, test_data,options.flag_reverse)

            test_loss, test_r2,test_mape,test_min_ratio,test_max_ratio = inference(model, test_data,options.batch_size,usage,save_file_dir,flag_save)
            print(
                '\ttest: loss={:.3f}\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(test_loss, test_r2,
                                                                                                         test_mape,test_min_ratio,test_max_ratio))

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
            if options.pretrain_dir is not None:
                model.load_state_dict(th.load(options.pretrain_dir,map_location='cuda:{}'.format(options.gpu)))

            model = model.to(device)

            print('seed:', seed)
            train(model)

    else:
        print('No checkpoint is specified. abandoning all model checkpoints and logs')
        model = init_model(options)
        if options.pretrain_dir is not None:
            model.load_state_dict(th.load(options.pretrain_dir,map_location='cuda:{}'.format(options.gpu)))

        model = model.to(device)
        train(model)