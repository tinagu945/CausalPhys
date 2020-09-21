import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from data.datasets import *


def update_ALIndexDataset(data, dataset, new_data_idx, which_nodes):
    """return: A new dataloader from a new dataset"""
    assert isinstance(dataset, ALIndexDataset)
    new_idx = torch.cat((dataset.idxs, new_data_idx))
    new_nodes = torch.cat((dataset.nodes, which_nodes))
    new_ds = ALIndexDataset(data, new_idx, new_nodes)
    return new_ds


def update_ALDataset(dataset, new_data, which_nodes, batch_size):
    """return: A new dataloader from a new dataset"""
    assert isinstance(dataset, ALDataset)
    data = torch.cat((dataset.data, new_data), dim=0)
    nodes = torch.cat((dataset.nodes, which_nodes), dim=0)
    new_ds = ALDataset(data, nodes, dataset.edge)
    return new_ds


def load_one_graph_data(suffix, train_data=None, control=False, self_loop=False, size=None, **kwargs):
    print('loading', suffix)
    feat = np.load('data/datasets/feat_'+suffix + '.npy')
    if size:
        feat = feat[:int(size)]
    edge = np.load('data/datasets/edges_' + suffix + '.npy')[0]

    mins, maxs = [], []
    num_atoms = feat.shape[1]

    # for i in range(6):
    #     print('before')
    #     print(set(feat[:, i, 0, 0]))
    # print('\n')
    if not train_data:
        # Normalize all 0th values to[-1, 1] (the following vector is one-hot indicating identity, so not normalized.)
        print('Using the dataset itself\'s max and min for normalization')
        for i in range(num_atoms):
            mmax = np.max(feat[:, i, :, 0])
            mmin = np.min(feat[:, i, :, 0])
            mins.append(mmin)
            maxs.append(mmax)
            feat[:, i, :, 0] = (feat[:, i, :, 0] - mmin)*2/(mmax-mmin)-1
    else:
        # Use train_data's max and min for normalization
        print('Using the train dataset\'s max and min for normalization')
        for i in range(num_atoms):
            mmax = np.max(feat[:, i, :, 0])
            mmin = np.min(feat[:, i, :, 0])
            mins.append(mmin)
            maxs.append(mmax)
            feat[:, i, :, 0] = (feat[:, i, :, 0] - train_data.mins[i]) * \
                2/(train_data.maxs[i]-train_data.mins[i])-1

    # for i in range(6):
    #     print(set(feat[:, i, 0, 0]))
    # print('\n')

    edge = np.reshape(edge, [-1, num_atoms ** 2])
    edge = np.array((edge + 1) / 2, dtype=np.int64)
    feat = torch.FloatTensor(feat)
    edge = torch.LongTensor(edge)
    # Exclude self edges
    if not self_loop:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
            [num_atoms, num_atoms])
    else:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_atoms, num_atoms))),
            [num_atoms, num_atoms])
    edge = edge[:, off_diag_idx]

    if control:
        dataset = ControlOneGraphDataset(feat, edge, mins, maxs, **kwargs)
    else:
        dataset = OneGraphDataset(feat, edge, mins, maxs)
    return dataset
