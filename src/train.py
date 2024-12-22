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
from torch.utils.data.sampler import SubsetRandomSampler
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
with open(os.path.join(options.data_savepath, 'ntype2id.pkl'), 'rb') as f:
    ntype2id,ntype2id_gate,ntype2id_module = pickle.load(f)
num_gate_types = len(ntype2id_gate)
num_gate_types -= 3
num_module_types = len(ntype2id_module)
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
# print(split_list)
# exit()
def load_data(usage):
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

        graph.ndata['feat'] = graph.ndata['ntype']
        graph.ndata['feat'] = graph.ndata['ntype'][:,3:]

        # print(th.sum(graph.ndata['value'][:,0])+th.sum(graph.ndata['value'][:,1])+th.sum(graph.ndata['is_pi'])+th.sum(graph.ndata['feat']),th.sum(graph.ndata['ntype']))

        graph.ndata['feat_module'] = graph.ndata['ntype_module']
        graph.ndata['feat_gate'] = graph.ndata['ntype_gate']
        graph_info['POs_feat'] = graph_info['POs_level_max'].unsqueeze(-1)
        graph.ndata['h'] = th.ones((graph.number_of_nodes(), options.hidden_dim), dtype=th.float)
        graph.ndata['PO_feat'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
        graph.ndata['PO_feat'][graph.ndata['is_po']==1] = graph_info['POs_feat']

        if len(graph_info['delay-label_pairs'][0][0])!= len(graph.ndata['is_pi'][graph.ndata['is_pi'] == 1]):
            print('skip',graph_info['design_name'])
            continue

        if options.flag_reverse:
            graph = add_reverse_edges(graph)
        if usage=='train' and options.flag_path_supervise:
            graph = add_newEtype(graph,'pi2po',([],[]),{})

        graph_info['graph'] = graph
        #graph_info['PI_mask'] = PI_mask
        #graph_info['delay-label_pairs'] = graph_info['delay-label_pairs'][1:]
        loaded_data.append(graph_info)

    batch_size = options.batch_size
    if usage=='test':
        batch_size = batch_size*4
    drop_last = True if usage == 'train' else False
    #drop_last = False
    sampler = SubsetRandomSampler(th.arange(len(loaded_data)))

    idx_loader = DataLoader([i for i in range(len(loaded_data))], sampler=sampler, batch_size=batch_size,
                              drop_last=drop_last)
    return loaded_data,idx_loader

def init_model(options):
    model = TimeConv(
            infeat_dim1=num_gate_types,
            infeat_dim2=num_module_types,
            hidden_dim=options.hidden_dim,
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

def test(model,test_data,test_idx_loader):

    model.flag_train = False
    with (th.no_grad()):
        total_num, total_loss, total_r2 = 0, 0.0, 0
        labels,labels_hat = None,None
        POs_topo = None

        for batch, idxs in enumerate(test_idx_loader):
            sampled_data = []
            idxs = idxs.numpy().tolist()
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
            if options.flag_reverse:
                topo_r = gen_topo(sampled_graphs, flag_reverse=True)
                graphs_info['topo_r'] = [l.to(device) for l in topo_r]
                nodes_list = th.tensor(range(sampled_graphs.number_of_nodes())).to(device)
                POs = nodes_list[sampled_graphs.ndata['is_po'] == 1]
                graphs_info['POs'] = POs.detach().cpu().numpy().tolist()
                # graph.ndata['hd'] = -1000*th.ones((graph.number_of_nodes(), len(POs)), dtype=th.float).to(device)
                sampled_graphs.ndata['hp'] = th.zeros((sampled_graphs.number_of_nodes(), len(POs)), dtype=th.float).to(device)
                for i, po in enumerate(POs):
                    sampled_graphs.ndata['hp'][po][i] = 1

            #num_cases = 2
            #print(data['design_name'])
            for i in range(num_cases):
                if num_cases==2 and i==0:
                    continue
                po_labels, pi_delays = None,None
                for data in sampled_data:
                    if options.target_residual:
                        PIs_delay,_, POs_label, pi2po_edges = data['delay-label_pairs'][i][:4]
                    else:
                        PIs_delay, POs_label, _,pi2po_edges  = data['delay-label_pairs'][i][:4]

                    graph = data['graph']
                    cur_po_labels =  th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    cur_po_labels[graph.ndata['is_po'] == 1] = th.tensor(POs_label, dtype=th.float).unsqueeze(-1)
                    cur_pi_delays = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
                    cur_pi_delays[graph.ndata['is_pi'] == 1] = th.tensor(PIs_delay, dtype=th.float).unsqueeze(1)
                    if po_labels is None:
                        po_labels = cur_po_labels
                        pi_delays = cur_pi_delays
                    else:
                        po_labels = th.cat((po_labels, cur_po_labels), dim=0)
                        pi_delays = th.cat((pi_delays, cur_pi_delays), dim=0)

                sampled_graphs.ndata['label'] = po_labels.to(device)
                sampled_graphs.ndata['delay'] = pi_delays.to(device)

                cur_labels = sampled_graphs.ndata['label'][graphs_info['POs_mask']].to(device)
                cur_labels_hat, path_loss = model(sampled_graphs, graphs_info)
                # cur_r2 = R2_score(cur_labels_hat, cur_labels).item()
                # cur_mape = th.mean(th.abs(cur_labels_hat[cur_labels != 0] - cur_labels[cur_labels != 0]) / cur_labels[cur_labels != 0])
                #print(graphs_info['design_name'])
                #print(i,"R2={:.2f} mape={:.3f}".format(cur_r2,cur_mape))
                #exit()
                if labels_hat is None:
                    labels_hat = cur_labels_hat
                    labels = cur_labels
                    POs_topo = POs_topolevel
                else:
                    labels_hat = th.cat((labels_hat, cur_labels_hat), dim=0)
                    labels = th.cat((labels, cur_labels), dim=0)
                    POs_topo = th.cat((POs_topo, POs_topolevel), dim=0)
            #exit()
        test_loss = Loss(labels_hat, labels).item()

        test_r2 = R2_score(labels_hat, labels).item()
        test_mape = th.mean(th.abs(labels_hat[labels != 0] - labels[labels != 0]) / labels[labels != 0])
        ratio = labels_hat[labels != 0] / labels[labels != 0]
        min_ratio = th.min(ratio)
        max_ratio = th.max(ratio)

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
        # plt.savefig('bar.png')
        # max_label = max(th.max(labels_hat).item(),th.max(labels).item())
        # plt.xlim(0, max_label)  # 设定绘图范围
        # plt.ylim(0, max_label)
        # plt.xlabel('predict')
        # plt.ylabel('label')
        # plt.scatter(labels_hat.detach().cpu().numpy().tolist(),labels.detach().cpu().numpy().tolist(),s=0.2)
        # plt.savefig('scatter.png')

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
        elif epoch % 3 ==2:
            optim = th.optim.Adam(
                itertools.chain(model.mlp_pi.parameters(), model.mlp_neigh_gate.parameters(),
                                model.mlp_neigh_module.parameters(), model.mlp_type.parameters(),
                                model.mlp_pos.parameters()), options.learning_rate, weight_decay=options.weight_decay
            )

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
            graphs_info['POs_mask'] = (sampled_graphs.ndata['is_po'] == 1).squeeze(-1).to(device)
            POs_topolevel = sampled_graphs.ndata['PO_feat'][sampled_graphs.ndata['is_po'] == 1].to(device)
            graphs_info['POs_feat'] = POs_topolevel
            graphs_info['topo'] = [l.to(device) for l in topo_levels]
            if options.flag_reverse:
                topo_r = gen_topo(sampled_graphs, flag_reverse=True)
                graphs_info['topo_r'] = [l.to(device) for l in topo_r]
                nodes_list = th.tensor(range(sampled_graphs.number_of_nodes())).to(device)
                POs = nodes_list[sampled_graphs.ndata['is_po'] == 1]
                graphs_info['POs'] = POs.detach().cpu().numpy().tolist()
                # graph.ndata['hd'] = -1000*th.ones((graph.number_of_nodes(), len(POs)), dtype=th.float).to(device)
                sampled_graphs.ndata['hp'] = th.zeros((sampled_graphs.number_of_nodes(), len(POs)), dtype=th.float).to(device)
                for i, po in enumerate(POs):
                    sampled_graphs.ndata['hp'][po][i] = 1
                    # graph.ndata['hd'][po][i] = 0

            for i in range(num_cases):
                po_labels, pi_delays = None,None
                start_idx = 0
                new_edges,new_edges_weight = ([],[]),[]
                for data in sampled_data:
                    if options.target_residual:
                        PIs_delay, _, POs_label,pi2po_edges = data['delay-label_pairs'][i][:4]
                    else:
                        PIs_delay, POs_label, _,pi2po_edges = data['delay-label_pairs'][i][:4]

                    graph = data['graph']
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

                sampled_graphs.ndata['label'] = po_labels.to(device)
                sampled_graphs.ndata['delay'] = pi_delays.to(device)

                labels_hat,path_loss = model(sampled_graphs, graphs_info)
                labels = sampled_graphs.ndata['label'][graphs_info['POs_mask']].to(device)

                total_num += len(labels)
                train_loss = Loss(labels_hat, labels)

                if options.flag_path_supervise:
                    #print(train_loss.item(),path_loss.item())
                    train_loss += -path_loss

                train_r2 = R2_score(labels_hat, labels).to(device)
                train_mape = th.mean(th.abs(labels_hat[labels!=0]-labels[labels!=0])/labels[labels!=0])
                ratio = labels_hat[labels!=0] / labels[labels!=0]
                min_ratio = th.min(ratio)
                max_ratio = th.max(ratio)
                if i==num_cases-1:
                    print('{}/{} train_loss:{:.3f}, {:.3f}\ttrain_r2:{:.3f}\ttrain_mape:{:.3f}, ratio:{:.2f}-{:.2f}'.format((batch+1)*options.batch_size,num_traindata,train_loss.item()+path_loss,-path_loss,train_r2.item(),train_mape.item(),min_ratio,max_ratio))

                optim.zero_grad()
                train_loss.backward()
                optim.step()
                torch.cuda.empty_cache()

                if options.flag_path_supervise:
                    sampled_graphs.remove_edges(sampled_graphs.edges('all',etype='pi2po')[2],etype='pi2po')

        torch.cuda.empty_cache()
        val_loss, val_r2,val_mape,val_min_ratio,val_max_ratio = test(model, val_data,val_idx_loader)
        test_loss, test_r2,test_mape,test_min_ratio,test_max_ratio = test(model,test_data,test_idx_loader)
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
        options.target_residual = input_options.target_residual
        #options.flag_filter = input_options.flag_filter
        #options.flag_reverse = input_options.flag_reverse
        #options.pi_choice = input_options.pi_choice
        options.batch_size = input_options.batch_size
        options.gpu = input_options.gpu
        options.flag_path_supervise = input_options.flag_path_supervise
        options.flag_reverse = input_options.flag_reverse
        options.pi_choice = input_options.pi_choice


        # print(options)
        # exit()
        model = init_model(options)
        model.flag_train = True

        # if options.flag_reverse:
        #     if options.pi_choice == 0: model.mlp_global_pi = MLP(2, int(options.hidden_dim / 2), options.hidden_dim)
        #     model.mlp_out_new = MLP(options.out_dim, options.hidden_dim, 1)
        model = model.to(device)
        model.load_state_dict(th.load(model_save_path,map_location='cuda:{}'.format(options.gpu)))
        test_data,test_idx_loader = load_data('test')
        test_loss, test_r2,test_mape,test_min_ratio,test_max_ratio = test(model, test_data,test_idx_loader)
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
            if options.flag_reverse:
                if options.pi_choice == 0: model.mlp_global_pi = MLP(2, int(options.hidden_dim / 2), options.hidden_dim)
                model.mlp_out_new =  MLP(options.out_dim,options.hidden_dim,1)
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