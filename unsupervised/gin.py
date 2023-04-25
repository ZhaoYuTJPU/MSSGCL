import numpy as np
import torch.nn
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv,global_add_pool
from tqdm import tqdm
from torch_geometric.utils import softmax
import torch_scatter


class Encoder(nn.Module):
    def __init__(self,num_features, dim, num_gc_layers, pooling):
        super(Encoder, self).__init__()

        self.pooling = pooling
        self.num_gc_layers = num_gc_layers
        self.pooling = pooling
        self.dim = dim

        self.fc = torch.nn.Linear(num_features, dim, bias=False)

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim,dim), ReLU(), Linear(dim,dim))
            else:
                nn = Sequential(Linear(dim,dim), ReLU(), Linear(dim,dim))
            conv = GINConv(nn)
            bn = BatchNorm1d(dim)
            act = ReLU()

            self.convs.append(conv)
            self.bns.append(bn)
            self.acts.append(act)
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0],1)).to(device)
        x = self.fc(x)
        # if node_imp is not None:
        #     out, _ = torch_scatter.scatter_max(torch.reshape(node_imp, (1, -1)), batch)
        #     out = out.reshape(-1, 1)
        #     out = out[batch]
        #     node_imp /= (out*10)
        #     node_imp += 0.9
        #     node_imp = node_imp.expand(-1, self.dim)
        xs = []
        for conv, act, bn in zip(self.convs, self.acts, self.bns):
            x = conv(x, edge_index)
            x = act(x)
            x = bn(x)
            # if node_imp is not None:
            #     x_imp = x * node_imp
            # else:
            #     x_imp = x
            xs.append(x)
        xpool = [global_add_pool(x, batch) for x in xs]
        if self.pooling == "last":
            x = xpool[-1]
        elif self.pooling == "all":
            x = torch.cat(xpool, 1)
        elif self.pooling == "add":
            x = 0.
            for layer in range(self.num_gc_layers):
                x += xpool[layer]
        return x, torch.cat(xs, 1)
    def get_embeddings(self,loader,device):
        ret = []
        y = []
        # loader = tqdm(loader,desc="---Eval---")
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x, _ = self.forward(x, edge_index, batch)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)

        return ret,y

class Translator(nn.Module):
    def __init__(self,num_features, dim, num_gc_layers):
        super().__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()

        for i in range(self.num_gc_layers):
            if i and i != self.num_gc_layers - 1:
                nn = Sequential(Linear(dim,dim),ReLU(),Linear(dim,dim))
                bn = BatchNorm1d(dim)
            elif i == num_gc_layers - 1:
                nn = Sequential(Linear(dim,dim),ReLU(),Linear(dim,1))
                bn = BatchNorm1d(1)
            else:
                nn = Sequential(Linear(num_features,dim),ReLU(),Linear(dim,dim))
                bn = BatchNorm1d(dim)
            act = ReLU()
            conv = GINConv(nn)

            self.convs.append(conv)
            self.bns.append(bn)
            self.acts.append(act)

    def forward(self,x,edge_index,batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)
        xs = []
        for i in range(self.num_gc_layers):
            if i != self.num_gc_layers - 1:
                x = self.convs[i](x,edge_index)
                x = self.bns[i](x)
                x = self.acts[i](x)
            else:
                x = self.convs[i](x,edge_index)
                x = self.bns[i](x)
            xs.append(x)
        node_prob = xs[-1]
        node_prob = softmax(node_prob/5.0,batch)

        return node_prob

class MLPHead(nn.Module):
    def __init__(self,in_channels, hidden_dim, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,out_channels)
        )
    def forward(self, x):
        return self.net(x)








