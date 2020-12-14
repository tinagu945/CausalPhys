import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset


class OneGraphDataset(Dataset):
    """One graph for all trajectories. Memory efficient."""

    def __init__(self, data, edge, mins, maxs):
        self.edge = edge
        self.data = data
        self.mins = mins
        self.maxs = maxs

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.edge

    @staticmethod
    def load_one_graph_data(suffix, train_data_min_max=None, control=False, self_loop=False, size=None, **kwargs):
        # print('loading', suffix)
        feat = np.load('data/datasets/feat_'+suffix + '.npy')
        if size:
            indices = np.arange(feat.shape[0])
            np.random.shuffle(indices)
            print('Dataset not controlled and has been shuffled', indices[:10])
            feat = feat[indices[:int(size)]]
        edge = np.load('data/datasets/edges_' + suffix + '.npy')[0]

        mins, maxs = [], []
        num_atoms = feat.shape[1]

        # for i in range(6):
        #     print('before')
        #     print(set(feat[:, i, 0, 0]))
        # print('\n')
        if not train_data_min_max:
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
                feat[:, i, :, 0] = (feat[:, i, :, 0] - train_data_min_max[0][i]) * \
                    2/(train_data_min_max[1][i]-train_data_min_max[0][i])-1

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


# For non-AL, use-all dataset sampler.
class ControlOneGraphDataset(Dataset):
    def __init__(self, data, edge, mins, maxs, control_nodes=5, variations=5, need_grouping=True):
        """Assume the nodes to be controlled are the top k nodes, and the target nodes are at bottom

        Args:
            data (np.array):
            edge (np.array): One graph for all trajectories. Memory efficient
            control_nodes (int): #nodes needs to be controlled, equals #nodes in data - #target nodes
            #values each node takes in the dataset. Currently needs be the same for all nodes. Defaults to 5.
            variations (int, optional):
            #variations of permutations. If no, data_size=dataset_size
            need_grouping: (bool, optional): Whether the data is actually of size dataset_size//variations. If so, the data must be arranged in
        """

        self.data = data
        self.edge = edge
        self.mins = mins
        self.maxs = maxs
        self.control_nodes = control_nodes
        self.variations = variations
        self.need_grouping = need_grouping

    def __len__(self):
        if self.need_grouping:
            return self.data.shape[0]*self.control_nodes
        else:
            return self.data.shape[0]

    def __getitem__(self, idx):
        if self.need_grouping:
            # data[j*(5**x)+i]
            # print(idx, self.data.shape[0])
            which_node, index = divmod(int(idx), self.data.shape[0])
            # import pdb
            # pdb.set_trace()
            which_node = self.control_nodes - which_node - 1
            # integer, reminder = divmod(a, b) = a/b
            # group size: self.variations**(which_node+1)
            which_group, group_offset = divmod(
                index, self.variations**(which_node+1))
            i, j = divmod(group_offset, self.variations)
            real_index = which_group*(self.variations**(which_node+1)) + \
                j*(self.variations**which_node)+i
            # print(which_node, index, which_group, group_offset, i, j, real_index)
            # reverse the order: which_node counts from zero from right changed to from left.
            which_node = self.control_nodes - which_node - 1
        else:
            real_index = int(idx)
            which_node, _ = divmod(real_index, int(
                self.data.shape[0]/self.control_nodes))
        return self.data[real_index], which_node, self.edge


# ALIndexDataset for dataset sampler
class ALIndexDataset(Dataset):
    """Data doesn't change, only indices to be changed.
    For fixed datset, using ALDataset is also ok
    but this is more memory efficient.
    """

    def __init__(self, dataset, data_idx, which_nodes):
        """[summary]

        Args:
            dataset ([type]): [description]
            data_idx (torch.Tensor): [description]
            which_nodes (torch.Tensor): [description]
            edge ([type]): [description]
        """
        assert isinstance(dataset, OneGraphDataset) or isinstance(
            dataset, ControlOneGraphDataset)
        self.dataset = dataset
        self.idxs = data_idx
        self.edge = self.dataset.edge
        self.nodes = which_nodes

    def __len__(self):
        return self.idxs.size(0)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]][0], self.nodes[idx], self.edge

    def update(self, new_data_idx, which_nodes):
        """return: A new dataloader from a new dataset"""
        self.idxs = torch.cat((self.idxs, new_data_idx))
        self.nodes = torch.cat((self.nodes, which_nodes))


# ALDataset for simulator sampler
# class ALDataset(Dataset):
#     """For data generated on the fly"""

#     def __init__(self, data, nodes, edge, mins, maxs):
#         self.nodes = nodes
#         self.dataset = data
#         self.edge = edge

#         # Normalize to [-1, 1]
#         num_atoms = data.size(1)
#         for i in range(num_atoms):
#             # TODO: should one-hot dimension be normalzed?
#             max_val = self.data[:, i, :, :].max()
#             min_val = self.data[:, i, :, :].min()
#             print('raw data each dimension max and min: ', i, max_val, min_val)
#             self.data[:, i, :, :] = (
#                 self.data[:, i, :, :] - min_val)*2/(max_val-min_val)-1

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, idx):
#         return self.data[idx], self.nodes[idx], self.edge

#     def update(self, new_data, which_nodes):
#         """return: A new dataloader from a new dataset"""
#         self.data = torch.cat((self.data, new_data), dim=0)
#         self.nodes = torch.cat((self.nodes, which_nodes), dim=0)


# ALDataset for simulator sampler
class RLDataset(Dataset):
    """For data generated on the fly"""

    def __init__(self, data, edge, mins, maxs):
        self.data = data
        self.edge = edge
        self.mins = mins
        self.maxs = maxs

        # Normalize the first dimension of feature to [-1, 1]
        self.num_atoms = len(self.mins)
        if len(self.data.size()) == 4:
            for i in range(self.num_atoms):
                self.data[:, i, :, 0] = (
                    self.data[:, i, :, 0] - self.mins[i])*2/(self.maxs[i]-self.mins[i])-1

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.edge

    def update(self, new_data):
        """return: A new dataloader from a new dataset"""
        for i in range(self.num_atoms):
            new_data[:, i, :, 0] = (
                new_data[:, i, :, 0] - self.mins[i])*2/(self.maxs[i]-self.mins[i])-1
        self.data = torch.cat((self.data, new_data), dim=0)
