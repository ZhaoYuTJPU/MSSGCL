import os
import os.path as osp
import shutil
import time

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data

from itertools import repeat, product
import numpy as np

from copy import deepcopy
import pdb


class TUDataset_aug(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = ('http://ls11-www.cs.tu-dortmund.de/people/morris/'
           'graphkerneldatasets')
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False, aug=None,global_size=0.8,local_size=0.2):
        self.name = name
        self.cleaned = cleaned
        super(TUDataset_aug, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
        if not (self.name == 'MUTAG' or self.name == 'PTC_MR' or self.name == 'DD' or self.name == 'PROTEINS' or self.name == 'NCI1' or self.name == 'NCI109'):
            edge_index = self.data.edge_index[0, :].numpy()
            _, num_edge = self.data.edge_index.size()
            nlist = [edge_index[n] + 1 for n in range(num_edge - 1) if edge_index[n] > edge_index[n + 1]]
            nlist.append(edge_index[-1] + 1)

            num_node = np.array(nlist).sum()
            self.data.x = torch.ones((num_node, 1))

            edge_slice = [0]
            k = 0
            for n in nlist:
                k = k + n
                edge_slice.append(k)
            self.slices['x'] = torch.tensor(edge_slice)

            '''
            print(self.data.x.size())
            print(self.slices['x'])
            print(self.slices['x'].size())
            assert False
            '''

        # self.node_score = torch.zeros(self.data["x"].shape[0],dtype=torch.half)
        self.global_size = global_size
        self.local_size = local_size
        self.aug = aug

    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url('{}/{}.zip'.format(url, self.name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get_num_feature(self):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[0]

        for key in self.data.keys:
            if key == 'num_nodes':
                continue
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[0],
                                                       slices[0 + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        _, num_feature = data.x.size()

        return num_feature


    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            if key == 'num_nodes':
                continue
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        """
        edge_index = data.edge_index
        node_num = data.x.size()[0]
        edge_num = data.edge_index.size()[1]
        data.edge_index = torch.tensor([[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if edge_index[0, n] < node_num and edge_index[1, n] < node_num] + [[n, n] for n in range(node_num)], dtype=torch.int64).t()
        """

        # node_num = data.edge_index.max()
        node_num = data.x.size()[0]
        # st = torch.min(data.edge_index)
        # en = torch.max(data.edge_index)
        # sl = torch.tensor([[n, n] for n in range(st,en + 1,1)]).t()
        sl = torch.tensor([[n,n] for n in range(node_num)]).t()
        data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        if self.aug == 'dnodes':
            data_aug_1 = drop_nodes(deepcopy(data),ratio=self.global_size)
            data_aug_2 = drop_nodes(deepcopy(data),ratio=self.global_size)
            data_aug_3 = drop_nodes(deepcopy(data),ratio=self.local_size)
            data_aug_4 = drop_nodes(deepcopy(data),ratio=self.local_size)
        elif self.aug == 'pedges':
            data_aug = permute_edges(deepcopy(data))
        elif self.aug == 'subgraph':
            data_aug_1 = subgraph(deepcopy(data),ratio=self.global_size)
            data_aug_2 = subgraph(deepcopy(data),ratio=self.global_size)
            data_aug_3 = subgraph(deepcopy(data),ratio=self.local_size)
            data_aug_4 = subgraph(deepcopy(data),ratio=self.local_size)
        elif self.aug == 'subgraph_prob':
            node_score = self.node_score[self.slices['x'][idx]:self.slices['x'][idx + 1]]
            data_aug_1 = subgraph_prob(deepcopy(data),ratio=self.global_size,node_score=node_score)
            data_aug_2 = subgraph_prob(deepcopy(data),ratio=self.global_size,node_score=node_score)
            data_aug_3 = subgraph_prob(deepcopy(data),ratio=self.local_size,node_score=node_score)
            data_aug_4 = subgraph_prob(deepcopy(data),ratio=self.local_size,node_score=node_score)
        elif self.aug == 'mask_nodes':
            data_aug = mask_nodes(deepcopy(data))
        elif self.aug == 'none':
            data_aug = data
            """
            if data.edge_index.max() > data.x.size()[0]:
                print(data.edge_index)
                print(data.x.size())
                assert False
            """
            # data_aug_global_1 = 1
            # data_aug_global_2 = 1
            # data_aug_local_1 = 1
            # data_aug_local_2 = 1
            # data_aug_global_1.x = torch.ones((data.edge_index.max()+1, 1))

        elif self.aug == 'random2':
            n = np.random.randint(2)
            if n == 0:
               data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
               data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False


        elif self.aug == 'random3':
            n = np.random.randint(3)
            if n == 0:
               data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
               data_aug = permute_edges(deepcopy(data))
            elif n == 2:
               data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False


        elif self.aug == 'random4':
            n = np.random.randint(4)
            if n == 0:
               data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
               data_aug = permute_edges(deepcopy(data))
            elif n == 2:
               data_aug = subgraph(deepcopy(data))
            elif n == 3:
               data_aug = mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False






        else:
            print('augmentation error')
            assert False

        # print(data, data_aug)
        # assert False
        if self.aug == "none":
            return data
        elif self.aug == "subgraph" or self.aug == 'subgraph_prob':
            return data,data_aug_1,data_aug_2,data_aug_3,data_aug_4


def drop_nodes(data,ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * (1 - ratio))

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    # idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()
    edge_index = edge_index.numpy()

    _, edge_num = edge_index.shape

    idx_not_missing = [n for n in range(node_num) if (n in edge_index[0] or n in edge_index[1])]
    node_num = len(idx_not_missing)
    data.x = data.x[idx_not_missing]

    idx_dict = {idx_not_missing[n]: n for n in range(node_num)}

    new_edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                      not edge_index[0, n] == edge_index[1, n]]
    if new_edge_index == []:
        new_edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num)]

    new_edge_index = torch.tensor(new_edge_index)
    new_edge_index.transpose_(0, 1)
    data.edge_index = new_edge_index

    # data.edge_index = edge_index

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def permute_edges(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num / 10)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]

    # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
    # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)
    edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data

def subgraph(data,ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * ratio)

    edge_index = data.edge_index.numpy()

    unique = np.unique(edge_index)
    idx_sub = [np.random.choice(list(unique), size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    while len(idx_sub) <= sub_num:
        idx_neigh = idx_neigh - idx_neigh.intersection(set(idx_sub))
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if set(idx_sub) == idx_neigh:
            break
        if sample_node in idx_sub:
            continue

        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))


    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()
    edge_index = edge_index.numpy()

    _, edge_num = edge_index.shape

    idx_not_missing = [n for n in range(node_num) if (n in edge_index[0] or n in edge_index[1])]
    node_num = len(idx_not_missing)
    data.x = data.x[idx_not_missing]

    idx_dict = {idx_not_missing[n]: n for n in range(node_num)}

    new_edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                      not edge_index[0, n] == edge_index[1, n]]
    if new_edge_index == []:
        new_edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num)]

    new_edge_index = torch.tensor(new_edge_index)
    new_edge_index.transpose_(0, 1)
    data.edge_index = new_edge_index

    return data


def mask_nodes(data):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 10)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

    return data

def subgraph_prob(data,ratio,node_score):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * ratio)

    node_prob = node_score.float()
    node_prob += 0.001
    node_prob = np.array(node_prob)
    node_prob /= node_prob.sum()

    edge_index = data.edge_index.numpy()

    node_list = np.unique(edge_index)
    idx_sub = [np.random.choice(list(node_list),replace=False,p=node_prob)]
    # idx_sub = [np.random.choice(list(unique), size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    while len(idx_sub) <= sub_num:
        idx_neigh = idx_neigh - idx_neigh.intersection(set(idx_sub))
        neigh_prob = np.array([node_prob[n] for n in idx_neigh])
        neigh_prob = neigh_prob / neigh_prob.sum()
        # neigh_prob = list(neigh_prob)
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh),replace=False,p=neigh_prob)
        if set(idx_sub) == idx_neigh:
            break
        if sample_node in idx_sub:
            continue

        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))


    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    # idx_nondrop = idx_sub

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()
    edge_index = edge_index.numpy()

    _, edge_num = edge_index.shape

    idx_not_missing = [n for n in range(node_num) if (n in edge_index[0] or n in edge_index[1])]
    node_num = len(idx_not_missing)
    data.x = data.x[idx_not_missing]

    idx_dict = {idx_not_missing[n]: n for n in range(node_num)}

    new_edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                      not edge_index[0, n] == edge_index[1, n]]
    if new_edge_index == []:
        new_edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num)]

    new_edge_index = torch.tensor(new_edge_index)
    new_edge_index.transpose_(0, 1)
    data.edge_index = new_edge_index

    return data

















