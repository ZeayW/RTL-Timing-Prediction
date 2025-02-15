import matplotlib.pyplot as plt
import torch
import dgl
print(dgl.__version__)
print(torch.__version__)

from model import PathModel
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

from options import get_options



options = get_options()
device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")
#device = th.device("cpu")
R2_score = R2Score().to(device)
Loss = nn.MSELoss()
Loss = nn.L1Loss()


data_path = options.data_savepath
if data_path.endswith('/'):
    data_path = data_path[:-1]
data_file = os.path.join(data_path, 'data.pkl')
#split_file = os.path.join(data_path, 'split.pkl')
split_file = os.path.join(os.path.split(data_path)[0], 'split_new.pkl')

with open(data_file, 'rb') as f:
    max_len,data_all = pickle.load(f)
    design_names = [d['design_name'].split('_')[-1] for d in data_all]

with open(split_file, 'rb') as f:
    split_list = pickle.load(f)


def cat_tensor(t1,t2):
    if t1 is None:
        return t2
    elif t2 is None:
        return t1
    else:
        return th.cat((t1,t2),dim=0)

def gather_data(data,index):

    random_paths = data['random_paths']
    critical_paths, pi2delay, POs_label = data['critical_path'][index]
    POs_feat = []
    for po_idx, rand_paths_info in enumerate(random_paths):
        feat_po = []
        critical_path_info = critical_paths[po_idx]
        feat_global = []
        feat_global.append(critical_path_info['rank'])
        feat_global.append(critical_path_info['rank_ratio'])
        # feat_global.append(rand_paths_info['num_nodes'])
        # feat_global.append(rand_paths_info['num_seq'])
        # feat_global.append(rand_paths_info['num_cmb'])
        feat_global.append(rand_paths_info['num_reg'])

        rand_paths = rand_paths_info['paths_rd']


        rand_paths.append(critical_path_info)
        for p in rand_paths:
            feat_path = []
            pi = p['path'][0]
            p_len = len(p['path'])
            input_delay = pi2delay[pi]
            feat_path.append(input_delay)
            feat_path.append(p_len)
            feat_path.extend(p['path_ntype'])
            #print('\t',len(feat_global),len(feat_path),len(p['path_degree']))
            feat_path.extend(p['path_degree'])

            feat_path.extend([0]*(max_len-p_len))

            feat = feat_global.copy()
            feat.extend(feat_path)
            feat_po.append(feat)
            #print('\t',len(feat))

        feat_po = th.tensor(feat_po,dtype=th.float).to(device)
        # print(feat_po.shape)
        POs_feat.append(feat_po)


    return POs_label,POs_feat




# print(split_list)
# exit()
def load_data(usage,flag_quick=True):

    assert usage in ['train','val','test']

    target_list = split_list[usage]
    target_list = [n.split('_')[-1] for n in target_list]

    dataset = [d for i,d in enumerate(data_all) if design_names[i] in target_list]
    case_range = (0, 100)
    if flag_quick:
        if usage == 'train':
            case_range = (0,20)
        else:
            case_range = (0, 40)
    print("------------Loading {}_data #{} {}-------------".format(usage,len(dataset),case_range))

    loaded_dataset = []
    for data in dataset:
        data['critical_path'] = data['critical_path'][case_range[0]:case_range[1]]
        loaded_dataset.append(data)

    return loaded_dataset

def get_idx_loader(data,batch_size):
    drop_last = True
    sampler = SubsetRandomSampler(th.arange(len(data)))
    idx_loader = DataLoader([i for i in range(len(data))], sampler=sampler, batch_size=batch_size,
                            drop_last=drop_last)
    return idx_loader


def init_model(options):
    # model = Graphormer(infeat_dim=num_gate_types+num_module_types+1,
    #                    feat_dim=128,
    #                    hidden_dim=256)

    model = PathModel(infeat_dim=14+max_len,hidden_dim=options.hidden_dim)
    print("creating model:")
    print(model)

    return model


def init(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



def test(model,test_data,batch_size):

    with (th.no_grad()):
        labels_all,labels_hat_all = None,None
        for i in range(0,len(test_data),batch_size):
            idxs = list(range(i,min(i+batch_size,len(test_data))))
            sampled_data = []
            num_cases = 100

            for idx in idxs:
                data = test_data[idx]
                num_cases = min(num_cases,len(data['critical_path']))
                shuffle(data['critical_path'])
                sampled_data.append(data)

            for i in range(num_cases):
                torch.cuda.empty_cache()
                labels = []
                POs_feat = []
                for data in sampled_data:
                    cur_labels,cur_feats = gather_data(data,i)
                    labels.extend(cur_labels)
                    POs_feat.extend(cur_feats)

                labels = th.tensor(labels, dtype=th.float).unsqueeze(1).to(device)
                labels_hat = model(POs_feat)

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

            for idx in idxs:
                data = train_data[idx]
                num_cases = min(num_cases,len(data['critical_path']))
                shuffle(data['critical_path'])
                sampled_data.append(data)

            total_labels,total_labels_hat = None,None

            #num_cases = 1
            for i in range(num_cases):
                torch.cuda.empty_cache()
                labels = []
                POs_feat = []
                for data in sampled_data:
                    cur_labels,cur_feats = gather_data(data,i)
                    labels.extend(cur_labels)
                    POs_feat.extend(cur_feats)
                labels = th.tensor(labels,dtype=th.float).unsqueeze(1).to(device)
                labels_hat = model(POs_feat)
                train_loss = Loss(labels_hat, labels)
                #print(train_loss)
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

                torch.cuda.empty_cache()
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

        model = init_model(options)

        model = model.to(device)
        model.load_state_dict(th.load(model_save_path,map_location='cuda:{}'.format(options.gpu)))
        usages = ['train','test','val']
        usages = ['test']
        for usage in usages:
            flag_save = True
            save_file_dir = options.checkpoint
            test_data = load_data(usage,options.quick)

            test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = test(model, test_data,options.batch_size)

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