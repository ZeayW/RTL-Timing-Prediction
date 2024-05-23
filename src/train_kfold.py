from options import get_options
from model import *
import pickle
import numpy as np
import os
from random import shuffle
import random
import torch as th
from torch.utils.data import DataLoader,Dataset
from torch.nn.functional import softmax
import torch.nn as nn
import datetime
from torchmetrics import R2Score
import tee

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

options = get_options()
device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")
R2_score = R2Score().to(device)
Loss = nn.MSELoss()

with open(os.path.join(options.data_savepath, 'ntype2id.pkl'), 'rb') as f:
    ntype2id = pickle.load(f)

def init_data(data):
    res = []
    for graph, graph_info in data:
        graph_info['topo'] = [t.to(device) for t in graph_info['topo']]
        graph_info['POs_feat'] = graph_info['POs_level_max'].unsqueeze(-1).to(device)
        # graph_info['POs_feat'] = th.cat([graph_info['POs_level_max'].unsqueeze(-1),graph_info['POs_level_min'].unsqueeze(-1)],dim=1).to(device)
        graph_info['POs_label'] = graph_info['POs_label'].to(device)
        graph.ndata['h'] = th.ones((graph.number_of_nodes(), options.hidden_dim), dtype=th.float)
        graph = graph.to(device)
        res.append((graph,graph_info))
    return res
def load_data_kfold(k):

    print("----------------Loading data----------------")
    data_path = options.data_savepath
    file = os.path.join(data_path, 'data.pkl')

    with open(file, 'rb') as f:
        data = pickle.load(f)

    data_kfold = []
    num_perfold = int(len(data) / k)
    shuffle(data)
    for i in range(k):
        data_train, data_test = [],[]
        for j in range(0,num_perfold * i):
            data_train.extend(data[j])
        for j in range(num_perfold * (i + 1),len(data)):
            data_train.extend([data[j]])
        for j in range(num_perfold * i,num_perfold * (i+1) ):
            data_test.extend(j)
        data_kfold.append((data_train, data_test))
    return data_kfold



def init_model(options):
    model = TimeConv(
            infeat_dim=len(ntype2id),
            hidden_dim=options.hidden_dim
        ).to(device)
    print("creating model:")
    print(model)

    return model


def init(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test(model,test_data):

    with (th.no_grad()):
        labels = None
        labels_hat = None
        for graph, graph_info in test_data:
            cur_labels = graph_info['POs_label']
            cur_labels_hat = model(graph, graph_info)
            if labels_hat is None:
                labels_hat = cur_labels_hat
                labels = cur_labels
            else:
                labels_hat = th.cat((labels_hat, cur_labels_hat), dim=0)
                labels = th.cat((labels, cur_labels), dim=0)
        labels = labels.unsqueeze(-1)
        val_loss = Loss(labels_hat, labels).item()
        val_r2 = R2_score(labels_hat, labels).item()

        return val_loss, val_r2

def train():
    print(options)
    th.multiprocessing.set_sharing_strategy('file_system')

    data_kfold = load_data_kfold(options.num_fold)

    print("Data successfully loaded")


    print("----------------Start training----------------")
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    best_r2s = []
    for fold, (data_train,data_test) in enumerate(data_kfold):
        model = init_model(options)
        model = model.to(device)
        optim = th.optim.Adam(
            model.parameters(), options.learning_rate, weight_decay=options.weight_decay
        )
        model.train()
        if options.checkpoint:
            model_save_dir = os.path.join('../checkpoints/{}'.format(options.checkpoint), 'f{}'.format(fold + 1))
            os.makedirs(model_save_dir, exist_ok=False)
        best_test_r2 = 0
        num_traindata = len(data_train)
        train_loader = DataLoader(list(range(len(data_train))), batch_size=options.batch_size, shuffle=True,
                                  drop_last=False)
        for epoch in range(options.num_epoch):
            print('Fold {}, Epoch {} ------------------------------------------------------------'.format(fold+1,epoch+1))
            total_num,total_loss, total_r2 = 0,0.0,0

            for batch, sample_indexs in enumerate(train_loader):
                labels = None
                labels_hat = None
                for idx in sample_indexs:
                    graph, graph_info = data_train[idx]
                    cur_labels = graph_info['POs_label']
                    cur_labels_hat = model(graph,graph_info)
                    if labels_hat is None:
                        labels_hat = cur_labels_hat
                        labels = cur_labels
                    else:
                        labels_hat = th.cat((labels_hat,cur_labels_hat), dim=0)
                        labels = th.cat((labels, cur_labels), dim=0)
                    # if embeddings is None:
                    #     embeddings = global_embedding.unsqueeze(0)
                    # else:
                    #     embeddings = th.cat((embeddings, global_embedding.unsqueeze(0)), dim=0)
                #print(embeddings.shape)
                labels = labels.unsqueeze(-1)
                total_num += len(labels)
                train_loss = Loss(labels_hat, labels)
                train_r2 = R2_score(labels_hat, labels).to(device)
                #print('loss:', train_loss.item())
                print('{}/{} train_loss:{:.5f}\ttrain_r2:{:.5f}'.format((batch+1)*options.batch_size,num_traindata,train_loss.item(),train_r2.item()))
                # total_loss += train_loss.item() * len(labels)
                # total_r2 += train_r2.item()
                optim.zero_grad()
                train_loss.backward()
                # print(model.GCN1.layers[0].attn_n.grad)
                optim.step()

            test_loss, test_r2 = test(model,data_test)
            best_test_r2 = max(best_test_r2,test_r2)
            print('Fold {}, end of {}, test_loss:{:.5f}\ttest_r2:{:.5f}'.format(fold+1,epoch,test_loss,test_r2))
            if options.checkpoint:
                th.save(model.state_dict(), os.path.join(model_save_dir,"{}.pth".format(epoch)))
                # print('\tsaved model to', os.path.join(save_path,"{}.pth".format(epoch)))
        best_r2s.append(best_test_r2)
    print("End all folds, r2:{}".format([round(v,5) for v in best_r2s]))

if __name__ == "__main__":
    seed = random.randint(1, 10000)
    init(seed)

    if options.checkpoint:
        print('saving logs and models to ../checkpoints/{}'.format(options.checkpoint))
        checkpoint_path = '../checkpoints/{}'.format(options.checkpoint)
        stdout_f = '../checkpoints/{}/stdout.log'.format(options.checkpoint)
        stderr_f = '../checkpoints/{}/stderr.log'.format(options.checkpoint)
        os.makedirs(checkpoint_path)  # exist not ok
        th.save(options, os.path.join(checkpoint_path, 'options.pkl'))
        with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
            print('seed:', seed)
            train()

    else:
        print('No checkpoint is specified. abandoning all model checkpoints and logs')
        train()