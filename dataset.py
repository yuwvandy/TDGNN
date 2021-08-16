import os.path as osp
from torch_geometric.datasets import Planetoid, WebKB, Actor
import torch_geometric.transforms as T
import torch
import numpy as np


def get_dataset(name, normalize_features = False, transform = None):
    if(name in ['Cora', 'Citeseer', 'Pubmed']):
        dataset = get_planetoid_dataset(name, normalize_features)
    elif(name in ['Wisconsin', 'Cornell', 'Texas']):
        dataset = get_WebKB_dataset(name, normalize_features)
    elif(name in ['Actor']):
        dataset = get_Actor_dataset(name, normalize_features)

    return dataset


def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset

def get_WebKB_dataset(name, normalize_features = False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = WebKB(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    dataset.data.y = dataset.data.y.long() #not sure why here the data.y is not of long type as planetoid

    return dataset

def get_Actor_dataset(name, normalize_features = False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Actor(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    dataset.data.y = dataset.data.y.long() #not sure why here the data.y is not of long type as planetoid

    return dataset

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype = torch.bool, device = index.device)
    mask[index] = 1

    return mask

def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data

def random_full_splits(data, num_classes, split, name):
    # Set new fixed random planetoid splits:

    indices = []
    train_indices = []
    val_indices = []
    test_indices = []

    spliter = np.load('./splits/' + name + '_split_0.6_0.2_' + str(split) + '.npz') #split contains 10 fixed full-supervised setting data

    data.train_mask = torch.tensor(spliter['train_mask']).type(torch.bool)
    data.val_mask = torch.tensor(spliter['val_mask']).type(torch.bool)
    data.test_mask = torch.tensor(spliter['test_mask']).type(torch.bool)


    return data
