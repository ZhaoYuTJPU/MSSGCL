import os
import os.path as osp
import random
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d, Sigmoid
from gin import Encoder,Translator
import numpy as np
from arguments import arg_parse
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from evaluate_embedding import evaluate_embedding
import time
import networkx as nx
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


class MSSGCL(nn.Module):
    def __init__(self,hidden_dim, num_gc_layers,pooling="all"):
        super(MSSGCL, self).__init__()

        if pooling == "last":
            self.embedding_dim = hidden_dim
        elif pooling == "all":
            self.embedding_dim = hidden_dim * num_gc_layers
        self.pooling = pooling
        self.translator = Translator(dataset_num_features,hidden_dim,num_gc_layers=3)
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers,pooling=self.pooling)
        self.project = nn.Sequential(
            Linear(self.embedding_dim,self.embedding_dim),
            ReLU(),
            Linear(self.embedding_dim,self.embedding_dim)
        )
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        y, M = self.encoder(x, edge_index, batch)
        y = self.project(y)
        return y

class Regressor(nn.Module):
    def __init__(self,input_dim,hidden_dim,out_dim=1,num_gc_layers=5,pooling="all"):
        super(Regressor, self).__init__()
        if pooling == "last":
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
        elif pooling == "all":
            self.input_dim = input_dim * num_gc_layers
            self.hidden_dim = hidden_dim * num_gc_layers
        self.out_dim = out_dim
        self.net = Sequential(
            Linear(self.input_dim,self.hidden_dim),
            ReLU(),
            BatchNorm1d(self.hidden_dim),
            Linear(self.hidden_dim, self.hidden_dim),
            ReLU(),
            BatchNorm1d(self.hidden_dim),
            Linear(self.hidden_dim, self.hidden_dim),
            ReLU(),
            BatchNorm1d(self.hidden_dim),
            Linear(self.hidden_dim, self.hidden_dim),
            ReLU(),
            BatchNorm1d(self.hidden_dim),
            Linear(self.hidden_dim, self.hidden_dim),
            ReLU(),
            BatchNorm1d(self.hidden_dim),
        )
        self.project = nn.Sequential(
            Linear(self.hidden_dim,self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim,1)
        )
        self.act = Sigmoid()

    def forward(self,x):
        x = self.net(x)
        x = self.project(x)
        x = self.act(x)
        return x

def info_nce(x1,x2):
    T = 0.2
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss

def info_nce_cross(global_1,global_2,local_1,local_2):
    return (info_nce(global_1,local_1) + info_nce(global_1,local_2) + info_nce(global_2,local_1) +info_nce(global_2,local_2)) / 4


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


# def graph_showing(data):
#     '''
#     args:
#          data: torch_geometric.data.Data
#     '''
#     G = nx.Graph()
#     edge_index = data['edge_index'].t()
#     #     print(edge_index)
#     edge_index = np.array(edge_index.cpu())
#     #     print(edge_index)
#
#     G.add_edges_from(edge_index)
#     nx.draw(G)
#     plt.show()




if __name__ == "__main__":
    args = arg_parse()
    setup_seed(args.seed)

    accuracies = {'val': [], 'test': []}
    epochs = args.epochs
    log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    num_workers = args.num_workers
    hidden_dim = args.hidden_dim
    num_gc_layers = args.num_gc_layers
    pooling = args.pooling
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', DS)

    dataset = TUDataset(path,name=DS,aug="subgraph",global_size=args.global_size,local_size=args.local_size).shuffle()
    # dataset = TUDataset(path, name=DS, aug="dnodes", global_size=args.global_size,
    #                     local_size=args.local_size).shuffle()
    dataset_eval = TUDataset(path,name=DS,aug="none").shuffle()

    print(len(dataset))
    print(dataset.get_num_feature())

    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=num_workers)
    dataloader_eval = DataLoader(dataset_eval,batch_size=batch_size,num_workers=0)
    # dataloader = tqdm(dataloader,desc="----Train----")
    # dataloader_eval = tqdm(dataloader_eval,desc="----Test----")

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    model = MSSGCL(hidden_dim=hidden_dim,num_gc_layers=num_gc_layers,pooling=pooling).to(device)
    regressor = Regressor(input_dim= 2 * hidden_dim,hidden_dim= 2 * 2 * hidden_dim,out_dim=1,num_gc_layers=num_gc_layers).to(device)
    optimizer = Adam(model.parameters(),lr=lr)
    optimizer_reg = Adam(regressor.parameters(),lr=lr)

    print('================')
    print('dataset: {}'.format(DS))
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('pooling: {}'.format(args.pooling))
    print('================')

    best_acc = 0.
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.
        for data in dataloader:
            data, data_aug_global_1, data_aug_global_2, data_aug_local_1, data_aug_local_2 = data
            data = data.to(device)
            data_aug_global_1 = data_aug_global_1.to(device)
            data_aug_global_2 = data_aug_global_2.to(device)
            data_aug_local_1 = data_aug_local_1.to(device)
            data_aug_local_2 = data_aug_local_2.to(device)
            # x = model(data.x, data.edge_index, data.batch)
            optimizer_reg.zero_grad()
            x = model(data.x, data.edge_index, data.batch)
            for bn in model.encoder.bns:
                bn.track_running_stats = False
            x_local_1 = model(data_aug_local_1.x, data_aug_local_1.edge_index, data_aug_local_1.batch)
            x_local_2 = model(data_aug_local_2.x, data_aug_local_2.edge_index, data_aug_local_2.batch)
            for bn in model.encoder.bns:
                bn.track_running_stats = True
            x_local_2_neg = x_local_2[torch.randperm(x_local_2.size(0))]
            x_local_mix = torch.cat([x_local_1,x_local_2],dim=1)
            x_local_mix_neg = torch.cat([x_local_1,x_local_2_neg],dim=1)
            # mix_score = regressor(x_local_mix).sum() / x_local_mix.size(0)
            mix_score = torch.mean(regressor(x_local_mix),dim=0)
            # mix_score_neg = regressor(x_local_mix_neg).sum() / x_local_2_neg.size(0)
            mix_score_neg = torch.mean(regressor(x_local_mix_neg),dim=0)
            loss_ll = -(mix_score - mix_score_neg)

            optimizer_reg.zero_grad()
            loss_ll.backward()
            optimizer_reg.step()

            # optimizer.zero_grad()
            for bn in model.encoder.bns:
                bn.track_running_stats = False
            x_local_1 = model(data_aug_local_1.x, data_aug_local_1.edge_index, data_aug_local_1.batch)
            x_local_2 = model(data_aug_local_2.x, data_aug_local_2.edge_index, data_aug_local_2.batch)
            for bn in model.encoder.bns:
                bn.track_running_stats = True
            for bn in model.encoder.bns:
                bn.track_running_stats = False
            x_global_1 = model(data_aug_global_1.x, data_aug_global_1.edge_index, data_aug_global_1.batch)
            x_global_2 = model(data_aug_global_2.x, data_aug_global_2.edge_index, data_aug_global_2.batch)
            for bn in model.encoder.bns:
                bn.track_running_stats = True
            x_local_mix = torch.cat([x_local_1,x_local_2],dim=1)

            loss_gg = info_nce(x_global_1,x_global_2)
            loss_gl = info_nce_cross(x_global_1,x_global_2,x_local_1,x_local_2)
            # loss_ll = regressor(x_local_mix).sum() / x_local_mix.size(0)
            loss_ll = torch.mean(regressor(x_local_mix),dim=0)

            loss = loss_gg + 0.8 * loss_gl + 0.4 * loss_ll

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        end = time.time()

        print('Epoch {}, Loss {:.4f}, Time {:.2f}s'.format(epoch, total_loss / len(dataloader),end - start))

        if epoch % log_interval == 0:
            start = time.time()
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval,device)
            acc_val, acc = evaluate_embedding(emb,y,device)
            if acc > best_acc:
                best_acc = acc
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            end = time.time()
            print("val_acc {:.2f}%, acc {:.2f}%, Time {:.2f}s".format(acc_val*100,acc*100,end-start))


    print(best_acc)

















