import torch
from torch.utils.data import Dataset

# For original datasets like val and test


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


# ALDataset for simulator sampler
class ALDataset(Dataset):
    """For data generated on the fly"""

    def __init__(self, data, nodes, edge):
        self.nodes = nodes
        self.data = data
        self.edge = edge

        # Normalize to [-1, 1]
        num_atoms = data.size(1)
        for i in range(num_atoms):
            # TODO: should one-hot dimension be normalzed?
            max_val = self.data[:, i, :, :].max()
            min_val = self.data[:, i, :, :].min()
            print('raw data each dimension max and min: ', i, max_val, min_val)
            self.data[:, i, :, :] = (
                self.data[:, i, :, :] - min_val)*2/(max_val-min_val)-1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.nodes[idx], self.edge
