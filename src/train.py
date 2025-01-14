import matplotlib.pyplot as plt
import torch

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




options = get_options()
device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")
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
split_file = os.path.join(os.path.split(data_path)[0], 'split.pkl')

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
def load_data(usage,flag_inference=False):
    assert usage in ['train','val','test']

    target_list = split_list[usage]
    target_list = [n.split('_')[-1] for n in target_list]
    #print(target_list[:10])

    data = [d for i,d in enumerate(data_all) if design_names[i] in target_list]
    print("------------Loading {}_data #{}-------------".format(usage,len(data)))

    loaded_data = []
    for  graph,graph_info in data:
        #print(graph_info['design_name'])
        #if int(graph_info['design_name'].split('_')[-1]) in [54, 96, 131, 300, 327, 334, 397]:
        #    continue
        name2nid = {graph_info['nodes_name'][i]:i for i in range(len(graph_info['nodes_name']))}

        if options.flag_homo:
            graph = heter2homo(graph)
        
        if options.inv_choice!=-1:
            graph.edges['intra_module'].data['is_inv'] = graph.edges['intra_module'].data['is_inv'].unsqueeze(1)
            graph.edges['intra_gate'].data['is_inv'] = graph.edges['intra_gate'].data['is_inv'].unsqueeze(1)
        graph.ndata['feat'] = graph.ndata['ntype']
        graph.ndata['feat'] = graph.ndata['ntype'][:,3:]

        # print(th.sum(graph.ndata['value'][:,0])+th.sum(graph.ndata['value'][:,1])+th.sum(graph.ndata['is_pi'])+th.sum(graph.ndata['feat']),th.sum(graph.ndata['ntype']))
        graph.ndata['width'] = graph.ndata['width'].unsqueeze(1)
        graph.ndata['feat_module'] = graph.ndata['ntype_module']
        graph.ndata['feat_gate'] = graph.ndata['ntype_gate'][:,3:]
        graph_info['POs_feat'] = graph_info['POs_level_max'].unsqueeze(-1)
        graph.ndata['h'] = th.ones((graph.number_of_nodes(), options.hidden_dim), dtype=th.float)
        graph.ndata['PO_feat'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
        graph.ndata['PO_feat'][graph.ndata['is_po']==1] = graph_info['POs_feat']

        if len(graph_info['delay-label_pairs'][0][0])!= len(graph.ndata['is_pi'][graph.ndata['is_pi'] == 1]):
            print('skip',graph_info['design_name'])
            continue

        if options.flag_reverse:
            graph = add_reverse_edges(graph)
        if  options.flag_path_supervise:
            graph = add_newEtype(graph,'pi2po',([],[]),{})

        graph_info['graph'] = graph
        #graph_info['PI_mask'] = PI_mask
        if options.quick and usage=='train':
            graph_info['delay-label_pairs'] = graph_info['delay-label_pairs'][:20]
        elif options.quick and usage!='train':
            graph_info['delay-label_pairs'] = graph_info['delay-label_pairs'][:40]
        loaded_data.append(graph_info)

    batch_size = options.batch_size
    if not flag_inference and (not options.flag_reverse or options.flag_path_supervise) and usage!='train':
        batch_size = len(loaded_data)


    print(batch_size)
    drop_last = True if usage == 'train' else False
    drop_last = False

    sampler = SubsetRandomSampler(th.arange(len(loaded_data)))

    idx_loader = DataLoader([i for i in range(len(loaded_data))], sampler=sampler, batch_size=batch_size,
                              drop_last=drop_last)
    return loaded_data,idx_loader

def init_model(options):
    model = TimeConv(
            infeat_dim1=num_gate_types,
            infeat_dim2=num_module_types,
            hidden_dim=options.hidden_dim,
            inv_choice= options.inv_choice,
            flag_width=options.flag_width,
            flag_delay_pd=options.flag_delay_pd,
            flag_delay_m=options.flag_delay_m,
            flag_delay_g=options.flag_delay_g,
            flag_delay_pi=options.flag_delay_pi,
            flag_ntype_g=options.flag_ntype_g,
            flag_path_supervise=options.flag_path_supervise,
            flag_filter = options.flag_filter,
            flag_reverse=options.flag_reverse,
            flag_splitfeat=options.split_feat,
            pi_choice=options.pi_choice,
            agg_choice=options.agg_choice,
            attn_choice=options.attn_choice,
            flag_homo=options.flag_homo,
            flag_global=options.flag_global,
            flag_attn=options.flag_attn
        ).to(device)
    print("creating model:")
    print(model)

    return model


def init(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def inference(model,test_data,test_idx_loader,usage='test',prob_file='',labels_file=''):

    new_dataset = []

    #model.flag_train = False
    with (th.no_grad()):
        total_num, total_loss, total_r2 = 0, 0.0, 0
        labels,labels_hat = None,None
        POs_criticalprob = None
        temp_labels, temp_labels_hat = None, None
        POs_topo = None

        batch_size = test_idx_loader.batch_size
        for i in range(0,len(test_data),batch_size):
            idxs = list(range(i,min(i+batch_size,len(test_data))))
        # for batch, idxs in enumerate(test_idx_loader):
        #     idxs = idxs.numpy().tolist()
            abnormal_POs = {}
            sampled_data = []
            num_cases = 100
            graphs = []
            for idx in idxs:
                data = test_data[idx]
                num_cases = min(num_cases,len(data['delay-label_pairs']))
                sampled_data.append(test_data[idx])
                graphs.append(data['graph'])
                #print(data['design_name'])

            if '00251' not in data['design_name']:
                continue

            sampled_graphs = dgl.batch(graphs)
            sampled_graphs = sampled_graphs.to(device)
            graphs_info = {}
            graphs_info = data
            topo_levels = gen_topo(sampled_graphs)
            graphs_info['is_heter'] = is_heter(sampled_graphs)
            graphs_info['topo'] = [l.to(device) for l in topo_levels]
            graphs_info['POs_mask'] = (sampled_graphs.ndata['is_po'] == 1).squeeze(-1).to(device)
            POs_topolevel = sampled_graphs.ndata['PO_feat'][sampled_graphs.ndata['is_po'] == 1].to(device)
            graphs_info['POs_feat'] = POs_topolevel
            if options.flag_reverse:
                topo_r = gen_topo(sampled_graphs, flag_reverse=True)
                graphs_info['topo_r'] = [l.to(device) for l in topo_r]
                nodes_list = th.tensor(range(sampled_graphs.number_of_nodes())).to(device)
                POs = nodes_list[sampled_graphs.ndata['is_po'] == 1]
                graphs_info['POs'] = POs.detach().cpu().numpy().tolist()
                sampled_graphs.ndata['hd'] = -1000*th.ones((sampled_graphs.number_of_nodes(), len(POs)), dtype=th.float).to(device)
                sampled_graphs.ndata['hp'] = th.zeros((sampled_graphs.number_of_nodes(), len(POs)), dtype=th.float).to(device)
                for k, po in enumerate(POs):
                    sampled_graphs.ndata['hp'][po][k] = 1
                    sampled_graphs.ndata['hd'][po][k] = 0

            new_delay_label_pairs = []

            # num_cases = 2
            print(data['design_name'])
            for j in range(num_cases):
                if j!=17:
                    continue
                # if num_cases==2 and i==0:
                #     continue
                po_labels,po_labels_margin, pi_delays = None,None,None
                start_idx = 0
                new_edges, new_edges_weight = ([], []), []
                for data in sampled_data:
                    if options.target_residual:
                        PIs_delay,POs_baselabel, POs_label, pi2po_edges = data['delay-label_pairs'][j][:4]
                    else:
                        PIs_delay, POs_label, POs_baselabel,pi2po_edges  = data['delay-label_pairs'][j][:4]


                    graph = data['graph']
                    if options.flag_path_supervise:
                        new_edges[0].extend([nid + start_idx for nid in pi2po_edges[0]])
                        new_edges[1].extend([nid + start_idx for nid in pi2po_edges[1]])
                        # new_edges_weight.extend(edges_weight)
                        start_idx += graph.number_of_nodes()

                    cur_po_labels =  th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    cur_po_labels[graph.ndata['is_po'] == 1] = th.tensor(POs_label, dtype=th.float).unsqueeze(-1)
                    # cur_po_labels_margin = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    # cur_po_labels_margin[graph.ndata['is_po'] == 1] = th.tensor(POs_baselabel, dtype=th.float).unsqueeze(-1)
                    cur_pi_delays = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    cur_pi_delays[graph.ndata['is_pi'] == 1] = th.tensor(PIs_delay, dtype=th.float).unsqueeze(1)

                    po_labels = cat_tensor(po_labels,cur_po_labels)
                    pi_delays = cat_tensor(pi_delays,cur_pi_delays)

                if options.flag_path_supervise:
                    # new_edges_feat = {'prob': th.tensor(new_edges_weight, dtype=th.float).unsqueeze(1)}
                    sampled_graphs.add_edges(th.tensor(new_edges[0]).to(device), th.tensor(new_edges[1]).to(device),
                                             etype='pi2po')

                sampled_graphs.ndata['label'] = po_labels.to(device)
                sampled_graphs.ndata['delay'] = pi_delays.to(device)
                sampled_graphs.ndata['input_delay'] = pi_delays.to(device)

                cur_labels = sampled_graphs.ndata['label'][graphs_info['POs_mask']].to(device)

                cur_labels_hat, path_loss,cur_POs_criticalprob = model(sampled_graphs, graphs_info)

                if options.flag_path_supervise:
                    sampled_graphs.remove_edges(sampled_graphs.edges('all', etype='pi2po')[2], etype='pi2po')


                labels_hat = cat_tensor(labels_hat,cur_labels_hat)
                labels = cat_tensor(labels,cur_labels)
                POs_criticalprob = cat_tensor(POs_criticalprob,cur_POs_criticalprob)

                if not options.flag_path_supervise:
                    continue

                abnormal_mask = (cur_POs_criticalprob.squeeze(1)<=0.05)
                normal_mask = (cur_POs_criticalprob.squeeze(1) >0.5)
                abnormal_POs = POs[abnormal_mask]
                normal_POs = POs[normal_mask]
                #print('\t',j,len(POs),len(abnormal_POs),len(normal_POs),'\t',round(th.mean(cur_POs_criticalprob).item(),2))
                #print(len(nodes_list),POs,abnormal_POs)

                #new_delay_label_pairs.append((PIs_delay, POs_label, POs_baselabel,pi2po_edges, abnormal_mask))
                data['delay-label_pairs'][j] = (PIs_delay, POs_label, POs_baselabel,pi2po_edges, abnormal_POs,normal_POs)


                mask = cur_POs_criticalprob.squeeze(1) <= 0.05
                nodes_list = th.tensor(range(sampled_graphs.number_of_nodes())).to(device)
                POs = nodes_list[sampled_graphs.ndata['is_po']==1]
                Pos_name = [data['nodes_name'][n] for n in POs]
                POs_low = POs[mask]
                POs_low_name = [data['nodes_name'][n] for n in POs_low]
                print(data['design_name'],j)
                print('\t',len(POs),len(POs_low),POs_low_name)
                # for po in POs_low_name:
                #     abnormal_POs[po] = abnormal_POs.get(po,0) + 1

            #print(abnormal_POs)
            # print(th.mean(POs_criticalprob))
            # if i>=5:
            #     exit()
            new_dataset.append((graph, data))

        # if options.flag_path_supervise:
        #     with open('new_data_{}2.pkl'.format(usage),'wb') as f:
        #         pickle.dump(new_dataset,f)

        test_loss = Loss(labels_hat, labels).item()

        test_r2 = R2_score(labels_hat, labels).item()
        test_mape = th.mean(th.abs(labels_hat[labels != 0] - labels[labels != 0]) / labels[labels != 0])
        ratio = labels_hat[labels != 0] / labels[labels != 0]
        min_ratio = th.min(ratio)
        max_ratio = th.max(ratio)

        # if not os.path.exists(prob_file):
        #     with open(prob_file,'wb') as f:
        #         pickle.dump(POs_criticalprob.detach().cpu().numpy().tolist(),f)
        # else:
        #     with open(prob_file, 'rb') as f:
        #         POs_criticalprob = pickle.load(f)
        #         POs_criticalprob = th.tensor(POs_criticalprob).to(device)

        mask1 = POs_criticalprob.squeeze(1) <= 0.05
        mask2 = POs_criticalprob.squeeze(1) > 0.5
        mask3 = th.logical_and(POs_criticalprob.squeeze(1) > 0.05,POs_criticalprob.squeeze(1) <=0.5)
        mask_l = labels.squeeze(1) != 0

        # if not os.path.exists(labels_file):
        #     with open(labels_file, 'wb') as f:
        #         pickle.dump(labels_hat[mask2].detach().cpu().numpy().tolist(), f)
        # else:
        #     with open(labels_file, 'rb') as f:
        #         labels_hat_high = pickle.load(f)
        #         labels_hat_high = th.tensor(labels_hat_high).to(device)
        #         labels_hat[mask2] = labels_hat_high
        print(th.mean(POs_criticalprob))
        print(len(labels[mask1]) / len(labels), len(labels[mask2]) / len(labels))
        temp_r2 = R2_score(labels_hat[mask1], labels[mask1]).item()
        temp_mape = th.mean(
            th.abs(labels_hat[th.logical_and(mask1, mask_l)] - labels[th.logical_and(mask1, mask_l)]) / labels[
                th.logical_and(mask1, mask_l)])
        print(temp_r2, temp_mape)
        temp_r2 = R2_score(labels_hat[mask2], labels[mask2]).item()
        temp_mape = th.mean(
            th.abs(labels_hat[th.logical_and(mask2, mask_l)] - labels[th.logical_and(mask2, mask_l)]) /
            labels[th.logical_and(mask2, mask_l)])
        print(temp_r2, temp_mape)

        temp_r3 = R2_score(labels_hat[mask3], labels[mask3]).item()
        temp_mape = th.mean(
            th.abs(labels_hat[th.logical_and(mask3, mask_l)] - labels[th.logical_and(mask3, mask_l)]) /
            labels[th.logical_and(mask3, mask_l)])
        print(temp_r3, temp_mape)
        # with open("labels2.pkl", 'wb') as f:
        #     pickle.dump(labels[mask3].detach().cpu(), f)
        # with open("labels2_hat.pkl", 'wb') as f:
        #     pickle.dump(labels_hat[mask3].detach().cpu(), f)

        # x = []
        # y = []
        # indexs = list(range(9,40))
        # for i,r in enumerate(indexs):
        #     r = r / 20
        #     if i==0:
        #         num = len(ratio[ratio<r+0.05])
        #     elif i== len(indexs)-1:
        #         num = len(ratio[ratio >= r])
        #     else:
        #         num = len(ratio[th.logical_and(ratio>=r,ratio<r+0.05)])
        #     x.append(r)
        #     y.append(num/len(ratio))
        # #plt.bar(x,y)
        # plt.xlabel('ratio')
        # plt.ylabel('percent')
        # plt.bar(x,y,width=0.03)
        # #print(list(zip(x,y)))
        #
        # plt.savefig('bar2.png')
        # max_label = max(th.max(labels_hat).item(),th.max(labels).item())
        # plt.xlim(0, max_label)  # 设定绘图范围
        # plt.ylim(0, max_label)
        # plt.xlabel('predict')
        # plt.ylabel('label')
        # plt.scatter(labels_hat.detach().cpu().numpy().tolist(),labels.detach().cpu().numpy().tolist(),s=0.2)
        # plt.savefig('scatter2.png')

        return test_loss, test_r2,test_mape,min_ratio,max_ratio
    #model.flag_train = True

def test(model,test_data,test_idx_loader):

    model.flag_train = False
    with (th.no_grad()):
        total_num, total_loss, total_r2 = 0, 0.0, 0
        labels,labels_hat = None,None

        batch_size = test_idx_loader.batch_size
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

            sampled_graphs = dgl.batch(graphs)
            sampled_graphs = sampled_graphs.to(device)
            graphs_info = {}
            topo_levels = gen_topo(sampled_graphs)
            graphs_info['is_heter'] = is_heter(sampled_graphs)
            graphs_info['topo'] = [l.to(device) for l in topo_levels]
            graphs_info['POs_mask'] = (sampled_graphs.ndata['is_po'] == 1).squeeze(-1).to(device)
            POs_topolevel = sampled_graphs.ndata['PO_feat'][sampled_graphs.ndata['is_po'] == 1].to(device)
            graphs_info['POs_feat'] = POs_topolevel
            if options.flag_reverse and not options.flag_path_supervise:
                topo_r = gen_topo(sampled_graphs, flag_reverse=True)
                graphs_info['topo_r'] = [l.to(device) for l in topo_r]
                nodes_list = th.tensor(range(sampled_graphs.number_of_nodes())).to(device)
                POs = nodes_list[sampled_graphs.ndata['is_po'] == 1]
                graphs_info['POs'] = POs.detach().cpu().numpy().tolist()
                sampled_graphs.ndata['hd'] = -1000*th.ones((sampled_graphs.number_of_nodes(), len(POs)), dtype=th.float).to(device)
                sampled_graphs.ndata['hp'] = th.zeros((sampled_graphs.number_of_nodes(), len(POs)), dtype=th.float).to(device)
                for k, po in enumerate(POs):
                    sampled_graphs.ndata['hp'][po][k] = 1
                    sampled_graphs.ndata['hd'][po][k] = 0

            # num_cases = 2
            # print(data['design_name'])
            flag_separate = False


            for j in range(num_cases):
                # if num_cases==2 and i==0:
                #     continue
                start_idx = 0
                new_POs = None
                po_labels,po_labels_margin, pi_delays = None,None,None
                for data in sampled_data:
                    if options.target_residual:
                        PIs_delay,POs_baselabel, POs_label, pi2po_edges = data['delay-label_pairs'][j][:4]
                    else:
                        PIs_delay, POs_label, POs_baselabel,pi2po_edges  = data['delay-label_pairs'][j][:4]

                    graph = data['graph']
                    num_nodes = graph.number_of_nodes()
                    if len(data['delay-label_pairs'][j])==6:
                        flag_separate = True
                        abnormal_POs,normal_POs = data['delay-label_pairs'][j][4:]

                        # nodes_list = th.tensor(range(graph.number_of_nodes())).to(device)
                        # POs = nodes_list[graph.ndata['is_po'] == 1]
                        # middle_POs = set(POs.detach().cpu().numpy().tolist()) - set(abnormal_POs.detach().cpu().numpy().tolist()) - set(normal_POs.detach().cpu().numpy().tolist())
                        # middle_POs = th.tensor(list(middle_POs),dtype=th.long).to(device)
                        # middle_POs =  middle_POs + start_idx
                        #print(len(POs),len(abnormal_POs),len(normal_POs),len(middle_POs))
                        #print(abnormal_POs,middle_POs)
                        abnormal_POs = abnormal_POs + start_idx
                        normal_POs = normal_POs + start_idx
                        start_idx += num_nodes
                        new_POs = cat_tensor(new_POs,normal_POs)


                    cur_po_labels =  th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    cur_po_labels[graph.ndata['is_po'] == 1] = th.tensor(POs_label, dtype=th.float).unsqueeze(-1)
                    # cur_po_labels_margin = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    # cur_po_labels_margin[graph.ndata['is_po'] == 1] = th.tensor(POs_baselabel, dtype=th.float).unsqueeze(-1)
                    cur_pi_delays = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    cur_pi_delays[graph.ndata['is_pi'] == 1] = th.tensor(PIs_delay, dtype=th.float).unsqueeze(1)

                    po_labels = cat_tensor(po_labels,cur_po_labels)
                    pi_delays = cat_tensor(pi_delays,cur_pi_delays)

                if flag_separate:
                    new_po_mask = th.zeros((sampled_graphs.number_of_nodes(),1),dtype=th.float)
                    new_po_mask[new_POs] = 1
                    new_po_mask = new_po_mask.squeeze(1).to(device)
                    #nodes_list = th.tensor(range(sampled_graphs.number_of_nodes())).to(device)
                    #shared_po = th.logical_and(sampled_graphs.ndata['is_po']==1, new_po_mask==1)
                    #print(len(new_POs),len(nodes_list[sampled_graphs.ndata['is_po']==1]),len(nodes_list[shared_po]))
                    sampled_graphs.ndata['is_po'] = new_po_mask
                    graphs_info['POs_mask'] = (sampled_graphs.ndata['is_po'] == 1).squeeze(-1).to(device)
                    graphs_info['POs'] = new_POs.detach().cpu().numpy().tolist()
                sampled_graphs.ndata['label'] = po_labels.to(device)
                sampled_graphs.ndata['delay'] = pi_delays.to(device)
                sampled_graphs.ndata['input_delay'] = pi_delays.to(device)

                cur_labels = sampled_graphs.ndata['label'][graphs_info['POs_mask']].to(device)

                cur_labels_hat, path_loss,_ = model(sampled_graphs, graphs_info)

                labels_hat = cat_tensor(labels_hat,cur_labels_hat)
                labels = cat_tensor(labels,cur_labels)

        # with open("labels2.pkl",'wb') as f:
        #     pickle.dump(labels.detach().cpu(),f)
        # with open("labels2_hat.pkl",'wb') as f:
        #     pickle.dump(labels_hat.detach().cpu(),f)


        test_loss = Loss(labels_hat, labels).item()

        test_r2 = R2_score(labels_hat, labels).item()
        test_mape = th.mean(th.abs(labels_hat[labels != 0] - labels[labels != 0]) / labels[labels != 0])
        ratio = labels_hat[labels != 0] / labels[labels != 0]
        min_ratio = th.min(ratio)
        max_ratio = th.max(ratio)

        return test_loss, test_r2,test_mape,min_ratio,max_ratio
    model.flag_train = True

def train(model):
    print(options)
    th.multiprocessing.set_sharing_strategy('file_system')

    train_data,train_idx_loader = load_data('train')
    val_data,val_idx_loader = load_data('val')
    test_data,test_idx_loader = load_data('test')
    print("Data successfully loaded")

    # set the optimizer
    # if options.flag_reverse and options.pi_choice==0:
    #     optim = th.optim.Adam(
    #          itertools.chain(model.mlp_global_pi.parameters(),model.mlp_out_new.parameters()),
    #          options.learning_rate, weight_decay=options.weight_decay
    #     )
    # elif options.flag_reverse and options.pi_choice==1:
    #     optim = th.optim.Adam(
    #         model.mlp_out.parameters(),
    #         options.learning_rate, weight_decay=options.weight_decay
    #     )
    # else:
    #     optim = th.optim.Adam(
    #         model.parameters(), options.learning_rate, weight_decay=options.weight_decay
    #     )
    model.train()


    print("----------------Start training----------------")

    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    num_traindata = len(train_data)
    for epoch in range(options.num_epoch):
        model.flag_train = True
        print('Epoch {} ------------------------------------------------------------'.format(epoch+1))
        total_num,total_loss, total_r2 = 0,0.0,0


        if options.flag_path_supervise:
            optim = th.optim.Adam(
                model.parameters(), options.learning_rate, weight_decay=options.weight_decay
            )
        elif not options.flag_reverse:
            optim = th.optim.Adam(
                model.parameters(), options.learning_rate, weight_decay=options.weight_decay
            )
        # elif epoch % 3 ==2:
        #     optim = th.optim.Adam(
        #         itertools.chain(model.mlp_pi.parameters(), model.mlp_neigh_gate.parameters(),
        #                         model.mlp_neigh_module.parameters(), model.mlp_type.parameters(),
        #                         model.mlp_pos.parameters()), options.learning_rate, weight_decay=options.weight_decay
        #     )

        elif options.pi_choice==0:
            optim = th.optim.Adam(
                itertools.chain(model.mlp_global_pi.parameters(), model.mlp_out_new.parameters()),
                options.learning_rate, weight_decay=options.weight_decay
            )

        elif options.pi_choice==1:
            optim = th.optim.Adam(
                model.mlp_out.parameters(),
                options.learning_rate, weight_decay=options.weight_decay
            )

        else:
            assert False


        for batch, idxs in enumerate(train_idx_loader):
            torch.cuda.empty_cache()
            sampled_data = []

            idxs = idxs.numpy().tolist()
            num_cases = 1000
            graphs = []

            for idx in idxs:
                data = train_data[idx]
                num_cases = min(num_cases,len(data['delay-label_pairs']))
                shuffle(train_data[idx]['delay-label_pairs'])
                sampled_data.append(train_data[idx])
                graphs.append(data['graph'])
            sampled_graphs = dgl.batch(graphs)
            sampled_graphs = sampled_graphs.to(device)
            topo_levels = gen_topo(sampled_graphs)
            graphs_info = {}
            graphs_info['is_heter'] = is_heter(sampled_graphs)

            POs_topolevel = sampled_graphs.ndata['PO_feat'][sampled_graphs.ndata['is_po'] == 1].to(device)
            graphs_info['POs_feat'] = POs_topolevel
            graphs_info['topo'] = [l.to(device) for l in topo_levels]
            if options.flag_reverse:
                topo_r = gen_topo(sampled_graphs, flag_reverse=True)
                graphs_info['topo_r'] = [l.to(device) for l in topo_r]
                nodes_list = th.tensor(range(sampled_graphs.number_of_nodes())).to(device)
                POs = nodes_list[sampled_graphs.ndata['is_po'] == 1]
                graphs_info['POs'] = POs.detach().cpu().numpy().tolist()
                # sampled_graphs.ndata['hd'] = -1000*th.ones((sampled_graphs.number_of_nodes(), len(POs)), dtype=th.float).to(device)
                # sampled_graphs.ndata['hp'] = th.zeros((sampled_graphs.number_of_nodes(), len(POs)), dtype=th.float).to(device)
                # for i, po in enumerate(POs):
                #     sampled_graphs.ndata['hp'][po][i] = 1
                #     sampled_graphs.ndata['hd'][po][i] = 0

            num_POs = 0
            totoal_path_loss = 0
            total_labels = None
            total_labels_hat = None
            flag_separate = False
            for i in range(num_cases):
                po_labels, pi_delays = None,None
                start_idx = 0
                new_edges,new_edges_weight = ([],[]),[]
                new_POs = None
                for data in sampled_data:
                    if options.target_residual:
                        PIs_delay, _, POs_label,pi2po_edges = data['delay-label_pairs'][i][:4]
                    else:
                        PIs_delay, POs_label, _,pi2po_edges = data['delay-label_pairs'][i][:4]

                    graph = data['graph']

                    if len(data['delay-label_pairs'][i])==6:
                        flag_separate = True
                        abnormal_POs,normal_POs = data['delay-label_pairs'][i][4:]
                        abnormal_POs = abnormal_POs + start_idx
                        normal_POs = normal_POs + start_idx
                        new_POs = cat_tensor(new_POs,normal_POs)

                    if options.flag_path_supervise:
                        new_edges[0].extend([nid+start_idx for nid in pi2po_edges[0]])
                        new_edges[1].extend([nid + start_idx for nid in pi2po_edges[1]])
                        #new_edges_weight.extend(edges_weight)

                    start_idx += graph.number_of_nodes()

                    cur_po_labels = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    cur_po_labels[graph.ndata['is_po'] == 1] = th.tensor(POs_label, dtype=th.float).unsqueeze(-1)
                    cur_pi_delays = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    cur_pi_delays[graph.ndata['is_pi'] == 1] = th.tensor(PIs_delay, dtype=th.float).unsqueeze(1)
                    if po_labels is None:
                        po_labels = cur_po_labels
                        pi_delays = cur_pi_delays
                    else:
                        po_labels = th.cat((po_labels, cur_po_labels), dim=0)
                        pi_delays = th.cat((pi_delays, cur_pi_delays), dim=0)

                if options.flag_path_supervise:
                    #new_edges_feat = {'prob': th.tensor(new_edges_weight, dtype=th.float).unsqueeze(1)}
                    sampled_graphs.add_edges(th.tensor(new_edges[0]).to(device), th.tensor(new_edges[1]).to(device),
                                    etype='pi2po')
                    #sampled_graphs = add_newEtype(sampled_graphs, 'pi2po', new_edges, {})


                # if new_POs is None or len(new_POs) ==0:
                #     continue
                if flag_separate:
                    new_po_mask = th.zeros((sampled_graphs.number_of_nodes(), 1), dtype=th.float)
                    new_po_mask[new_POs] = 1
                    new_po_mask = new_po_mask.squeeze(1).to(device)

                    #nodes_list = th.tensor(range(sampled_graphs.number_of_nodes())).to(device)
                    #shared_po = th.logical_and(sampled_graphs.ndata['is_po']==1, new_po_mask==1)
                    #print(len(new_POs),len(nodes_list[sampled_graphs.ndata['is_po']==1]),len(nodes_list[shared_po]))
                    sampled_graphs.ndata['is_po'] = new_po_mask
                graphs_info['POs_mask'] = (sampled_graphs.ndata['is_po'] == 1).squeeze(-1).to(device)

                sampled_graphs.ndata['label'] = po_labels.to(device)
                sampled_graphs.ndata['delay'] = pi_delays.to(device)
                sampled_graphs.ndata['input_delay'] = pi_delays.to(device)

                if options.flag_reverse:
                    if flag_separate:
                        POs = new_POs
                        graphs_info['POs'] = POs.detach().cpu().numpy().tolist()
                    sampled_graphs.ndata['hd'] = -1000*th.ones((sampled_graphs.number_of_nodes(), len(POs)), dtype=th.float).to(device)
                    sampled_graphs.ndata['hp'] = th.zeros((sampled_graphs.number_of_nodes(), len(POs)), dtype=th.float).to(device)
                    for j, po in enumerate(POs):
                        sampled_graphs.ndata['hp'][po][j] = 1
                        sampled_graphs.ndata['hd'][po][j] = 0

                labels_hat,path_loss,_ = model(sampled_graphs, graphs_info)
                labels = sampled_graphs.ndata['label'][graphs_info['POs_mask']].to(device)
                #print(len(labels))
                total_num += len(labels)
                train_loss = 0
                train_loss = Loss(labels_hat, labels)

                num_POs += len(path_loss)
                totoal_path_loss += th.sum(path_loss).item()
                total_labels = cat_tensor(total_labels,labels)
                total_labels_hat = cat_tensor(total_labels_hat, labels_hat)

                #print(len(labels),len(path_loss))
                path_loss = th.mean(path_loss)


                if options.flag_path_supervise:
                    #print(train_loss.item(),path_loss.item())
                    train_loss += -path_loss
                    #train_loss = th.exp(1-path_loss)*train_loss
                    #train_loss = th.mean(th.exp(1 - path_loss) * th.abs(labels_hat-labels))
                    pass


                if i==num_cases-1:
                    train_r2 = R2_score(total_labels_hat, total_labels).to(device)
                    train_mape = th.mean(th.abs(total_labels_hat[total_labels != 0] - total_labels[total_labels != 0]) / total_labels[total_labels != 0])
                    ratio = total_labels_hat[total_labels != 0] / total_labels[total_labels != 0]
                    min_ratio = th.min(ratio)
                    max_ratio = th.max(ratio)
                    path_loss_avg = totoal_path_loss / num_POs
                    #print(data['design_name'],len(total_labels),num_POs)
                    print('{}/{} train_loss:{:.3f}, {:.3f}\ttrain_r2:{:.3f}\ttrain_mape:{:.3f}, ratio:{:.2f}-{:.2f}'.format((batch+1)*options.batch_size,num_traindata,train_loss.item(),path_loss_avg,train_r2.item(),train_mape.item(),min_ratio,max_ratio))

                if len(labels) ==0:
                    continue

                optim.zero_grad()
                train_loss.backward()
                optim.step()
                torch.cuda.empty_cache()

                if options.flag_path_supervise:
                    sampled_graphs.remove_edges(sampled_graphs.edges('all',etype='pi2po')[2],etype='pi2po')

        torch.cuda.empty_cache()
        model.flag_train = False
        val_loss, val_r2,val_mape,val_min_ratio,val_max_ratio = test(model, val_data,val_idx_loader)
        test_loss, test_r2,test_mape,test_min_ratio,test_max_ratio = test(model,test_data,test_idx_loader)
        model.flag_train = True
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
        print(options)
        assert options.checkpoint, 'no checkpoint dir specified'
        model_save_path = '../checkpoints/{}/{}.pth'.format(options.checkpoint, options.test_iter)
        assert os.path.exists(model_save_path), 'start_point {} of checkpoint {} does not exist'.\
            format(options.test_iter, options.checkpoint)
        input_options = options
        options = th.load('../checkpoints/{}/options.pkl'.format(options.checkpoint))
        options.data_savepath = input_options.data_savepath
        options.target_residual = input_options.target_residual
        #options.flag_filter = input_options.flag_filter
        #options.flag_reverse = input_options.flag_reverse
        #options.pi_choice = input_options.pi_choice
        options.batch_size = input_options.batch_size
        options.gpu = input_options.gpu
        options.flag_path_supervise = input_options.flag_path_supervise
        options.flag_reverse = input_options.flag_reverse
        options.pi_choice = input_options.pi_choice
        options.quick = input_options.quick
        options.flag_delay_pd = input_options.flag_delay_pd
        options.inv_choice = input_options.inv_choice

        # print(options)
        # exit()
        model = init_model(options)
        model.flag_train = True
        flag_inference = True

        #if True:
        if options.flag_reverse and not options.flag_path_supervise:
            if options.pi_choice == 0: model.mlp_global_pi = MLP(2, int(options.hidden_dim / 2), options.hidden_dim)
            model.mlp_out_new = MLP(options.out_dim, options.hidden_dim, 1)
        model = model.to(device)
        model.load_state_dict(th.load(model_save_path,map_location='cuda:{}'.format(options.gpu)))
        usages = ['train','test','val']
        usages = ['test']
        for usage in usages:
            save_file_dir = '../checkpoints/cases_round6_v2/heter_filter_fixmux_fixbuf_simplify01_merge_fixPO_fixBuf/bs32_attn0-smoothmax_noglobal_piFeatValue_reduceNtype_featpos_mse_attnLeaky02_pathsum_init'
            #save_file_dir = options.checkpoint
            prob_file = os.path.join(save_file_dir,'POs_criticalprob_{}2.pkl'.format(usage))
            labels_file = os.path.join(save_file_dir,'labels_hat_high3.pkl')

            test_data,test_idx_loader = load_data(usage,flag_inference)

            test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = test(model, test_data,test_idx_loader)

            #test_loss, test_r2,test_mape,test_min_ratio,test_max_ratio = inference(model, test_data,test_idx_loader,usage,prob_file,labels_file)
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
            # if options.pretrain_dir is not None:
            #     model.load_state_dict(th.load(options.pretrain_dir,map_location='cuda:{}'.format(options.gpu)))

            if options.flag_reverse and not options.flag_path_supervise:
                if options.pi_choice == 0:
                    model.mlp_global_pi = MLP(2, int(options.hidden_dim / 2), options.hidden_dim)
                model.mlp_out_new =  MLP(options.out_dim,options.hidden_dim,1)
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
        if options.flag_reverse:
            if options.pi_choice == 0: model.mlp_global_pi = MLP(2, int(options.hidden_dim / 2), options.hidden_dim)
            model.mlp_out_new = MLP(options.out_dim, options.hidden_dim, 1)
        model = model.to(device)
        train(model)