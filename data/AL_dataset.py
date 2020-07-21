import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader


#TODO: make edges save space for AL and other scripts. Best would be only one copy of edge for all trajectories.
class ALIndexDataset(Dataset):
    """Data doesn't change, only indices to be changed"""
    def __init__(self, data, data_idx, edges, which_nodes):
        self.existing_idxs = existing_idxs
        self.existing_edges = edges
        self.existing_nodes = which_nodes
        self.data = data

    def __len__(self):
        return self.data_idx.shape[0]

    def __getitem__(self, idx):
        
        return self.data[self.existing_[idx]], self.existing_edges[0], self.existing_nodes[idx]
    

def update_ALIndexDataset(dataset, new_data_idx, which_nodes):
    """return: A new dataloader from a new dataset"""
    assert isinstance(dataset, ALIndexDataset)
    new_idx = torch.cat((dataset.existing_idxs, new_data_idx), dim=-1)
    new_nodes = torch.cat((dataset.existing_nodes, new_data_idx), dim=-1)
    new_ds = ALIndexDataset(new_idx, new_nodes, edges)
    new_dataloader = DataLoader(new_ds, batch_size=variations, shuffle=False)
    
    return new_ds, new_dataloader    




class ALDataset(Dataset):
    def __init__(self, data, nodes):
        self.nodes = nodes
        self.data = data
        
        # Normalize to [-1, 1]
        num_atoms = data.size(1)
        for i in range(num_atoms):
            #TODO: should one-hot dimension be normalzed?
            max_val =  self.data[:,i,:,:].max()   
            min_val =  self.data[:,i,:,:].min()      
            print('raw data each dimension max and min: ', i, max_val, min_val)
            self.data[:,i,:,:] = (self.data[:,i,:,:] - min_val)*2/(max_val-min_val)-1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):   
        return self.data[idx], self.nodes[idx]
      
    
def update_ALDataset(dataset, new_data, which_nodes, batch_size):
    """return: A new dataloader from a new dataset"""
    assert isinstance(dataset, ALDataset)
    data = torch.cat((dataset.data, new_data), dim=0)
    nodes = torch.cat((dataset.nodes, which_nodes), dim=0)
#     import pdb;pdb.set_trace()
    new_ds = ALDataset(data, nodes)
    new_dataloader = DataLoader(new_ds, batch_size=batch_size, shuffle=False)
    return new_ds, new_dataloader            



class OneGraphDataset(Dataset):
    """One graph for all trajectories. Memory efficient."""
    def __init__(self, data, edge):
        self.edge = edge
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):   
        return self.data[idx], self.edge
    
    

def load_AL_data(batch_size=1, suffix='_my', self_loop=False, total_size=None):
    """Only returns dataloaders for valid and test"""
    feat_valid = np.load('data/datasets/feat_valid' + suffix + '.npy')[:int(total_size[1])]
    edges_valid = np.load('data/datasets/edges_valid' + suffix + '.npy')[0]

    feat_test = np.load('data/datasets/feat_test' + suffix + '.npy')[:int(total_size[2])]
    edges_test = np.load('data/datasets/edges_test' + suffix + '.npy')[0]
    print(feat_valid.shape, feat_test.shape)

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = feat_valid.shape[1]

    for i in range(0, num_atoms):
#         print(i, np.max(feat_valid[:,i,:,:]), np.min(feat_valid[:,i,:,:]))
        feat_valid[:,i,:,:] = (feat_valid[:,i,:,:] - np.min(feat_valid[:,i,:,:]))*2/\
        (np.max(feat_valid[:,i,:,:])-np.min(feat_valid[:,i,:,:]))-1
#         print(i, np.max(feat_test[:,i,:,:]), np.min(feat_test[:,i,:,:]))
        feat_test[:,i,:,:] = (feat_test[:,i,:,:] - np.min(feat_test[:,i,:,:]))*2/\
        (np.max(feat_test[:,i,:,:])-np.min(feat_test[:,i,:,:]))-1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    if not self_loop:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
            [num_atoms, num_atoms])
    else:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_atoms, num_atoms))),
            [num_atoms, num_atoms])
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    valid_data = OneGraphDataset(feat_valid, edges_valid)
    test_data = OneGraphDataset(feat_test, edges_test)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return valid_data_loader, test_data_loader




def load_my_data(batch_size=1, suffix='_my', self_loop=False, total_size=None, control=False, \
                 control_nodes=None, variations=4, control_batch_size=4):
    feat_train = np.load('data/feat_train' + suffix + '.npy')[:int(total_size[0])]
    edges_train = np.load('data/edges_train' + suffix + '.npy')[:int(total_size[0])]

    feat_valid = np.load('data/feat_valid' + suffix + '.npy')[:int(total_size[1])]
    edges_valid = np.load('data/edges_valid' + suffix + '.npy')[:int(total_size[1])]

    feat_test = np.load('data/feat_test' + suffix + '.npy')[:int(total_size[2])]
    edges_test = np.load('data/edges_test' + suffix + '.npy')[:int(total_size[2])]
    print(feat_train.shape, feat_test.shape)

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = feat_train.shape[1]

#     loc_max = loc_train.max()
#     loc_min = loc_train.min()
#     vel_max = vel_train.max()
#     vel_min = vel_train.min()

#     # Normalize to [-1, 1]
#     loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
#     vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

#     loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
#     vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

#     loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
#     vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1
    for i in range(0, feat_train.shape[1]):
#         print(i, np.max(feat_train[:,i,:,:]), np.min(feat_train[:,i,:,:]))
        feat_train[:,i,:,:] = (feat_train[:,i,:,:] - np.min(feat_train[:,i,:,:]))*2/\
        (np.max(feat_train[:,i,:,:])-np.min(feat_train[:,i,:,:]))-1
#         print(i, np.max(feat_valid[:,i,:,:]), np.min(feat_valid[:,i,:,:]))
        feat_valid[:,i,:,:] = (feat_valid[:,i,:,:] - np.min(feat_valid[:,i,:,:]))*2/\
        (np.max(feat_valid[:,i,:,:])-np.min(feat_valid[:,i,:,:]))-1
#         print(i, np.max(feat_test[:,i,:,:]), np.min(feat_test[:,i,:,:]))
        feat_test[:,i,:,:] = (feat_test[:,i,:,:] - np.min(feat_test[:,i,:,:]))*2/\
        (np.max(feat_test[:,i,:,:])-np.min(feat_test[:,i,:,:]))-1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    if not self_loop:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
            [num_atoms, num_atoms])
    else:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_atoms, num_atoms))),
            [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]
#     import pdb;pdb.set_trace()
    if not control:
        train_data = TensorDataset(feat_train, edges_train)
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    else:
        train_data = ControlDataset(feat_train, edges=edges_train, control_nodes=control_nodes, variations=variations)
        train_data_loader = DataLoader(train_data, batch_size=control_batch_size, shuffle=False)
    
#     import pdb;pdb.set_trace()
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, valid_data_loader, test_data_loader

