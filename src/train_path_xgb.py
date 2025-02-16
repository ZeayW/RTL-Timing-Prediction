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
import pandas as pd
# from preprocess import *
import xgboost as xgb



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
        feat = []
        # feat.append(critical_path_info['rank'])
        # feat.append(critical_path_info['rank_ratio'])
        feat.append(rand_paths_info['num_nodes'])
        # feat.append(rand_paths_info['num_seq'])
        # feat.append(rand_paths_info['num_cmb'])
        #feat.append(rand_paths_info['num_reg'])
        path = critical_path_info['path']
        pi = path[0]
        input_delay = pi2delay[pi]
        path_len = len(path)
        feat.append(input_delay)
        feat.append(path_len)
        feat.extend(critical_path_info['path_ntype'])
        feat.extend(critical_path_info['path_degree'])
        feat.extend([0] * (max_len - path_len))

        # print(feat_po.shape)
        POs_feat.append(feat)


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

    labels, feat = [],[]

    for data in dataset:
        for i in range(case_range[0],case_range[1]):
            cur_label,cur_feat = gather_data(data,i)
            labels.extend(cur_label)
            feat.extend(cur_feat)

    feat = np.array(feat)
    labels = np.array(labels)
    return feat,labels



def init(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



def test(usage,model):
    feat, labels = load_data(usage, options.quick)
    labels_hat = model.predict(feat)
    labels = th.tensor(labels).ti(device)
    labels_hat = th.tensor(labels_hat).to(device)
    test_r2 = R2_score(labels_hat, labels).item()
    test_mape = th.mean(th.abs(labels_hat[labels != 0] - labels[labels != 0]) / labels[labels != 0])
    ratio = labels_hat[labels != 0] / labels[labels != 0]
    min_ratio = th.min(ratio)
    max_ratio = th.max(ratio)

    return test_r2,test_mape,min_ratio,max_ratio


def train():
    print(options)
    th.multiprocessing.set_sharing_strategy('file_system')

    train_feat,train_label = load_data('train',options.quick)

    train_feat = pd.DataFrame(train_feat)
    train_label = pd.DataFrame(train_label)
    print('Training ...')
    #xgbr = xgb.XGBRegressor(n_estimators=100, max_depth=45, nthread=25)
    xgbr = xgb.XGBRegressor(n_estimators=45, max_depth=8, nthread=25)
    xgbr.fit(train_feat, train_label)

    save_dir = '../checkpoints/{}'.format(options.checkpoint)
    with open(f"{save_dir}/ep_model.pkl", "wb") as f:
        pickle.dump(xgbr, f)
    print('Finish!')

    print('Testing ...')
    test('test',xgbr)

if __name__ == "__main__":
    seed = random.randint(1, 10000)
    init(seed)
    if options.test_iter:

        assert options.checkpoint, 'no checkpoint dir specified'
        model_save_path = '../checkpoints/{}/ep_model.pkl'.format(options.checkpoint)
        assert os.path.exists(model_save_path)

        with open(model_save_path,'rb') as f:
            model = pickle.load(f)

        usages = ['train','test','val']
        usages = ['test']
        for usage in usages:
            test_r2, test_mape, test_min_ratio, test_max_ratio = test(usage,model)
            print(
                '\ttest:\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(test_r2,
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
            train()

    else:
        print('No checkpoint is specified. abandoning all model checkpoints and logs')

        train()