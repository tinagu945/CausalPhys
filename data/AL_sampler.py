import torch
from torch.distributions import Categorical
import numpy as np
from torch.utils.data.sampler import Sampler
from data.datasets import *


class RandomPytorchSampler(Sampler):
    """
    Samples elements randomly as groups.
    """

    def __init__(self, dataset):
        assert isinstance(
            dataset, ControlOneGraphDataset), "To use dataset sampler, the dataset must be controlled!"
        self.dataset = dataset
        num_groups = int(len(self.dataset)/self.dataset.variations)
        self.group_order = np.random.permutation(num_groups)
        self.datapoints_order = []
        for i in self.group_order:
            self.datapoints_order.extend(
                np.arange(i*self.dataset.variations, (i+1)*self.dataset.variations))
        self.datapoints_order = torch.Tensor(self.datapoints_order).cuda()

    def __iter__(self):
        return iter(self.datapoints_order)

    def __len__(self):
        return len(self.dataset)


class AbstractSampler(object):
    def __init__(self):
        return NotImplementedError()

    def criterion(self, dist):
        return NotImplementedError()

    def sample(self, dist, batch_size):
        return NotImplementedError()


class MaximalEntropyDatasetSampler(AbstractSampler):
    def __init__(self, dataset):
        assert isinstance(
            dataset, ControlOneGraphDataset), "To use dataset sampler, the dataset must be controlled!"
        self.dataset = dataset
        num_groups = int(self.dataset.data.shape[0]/self.dataset.variations)
        self.datapoints_order = {}
        for i in range(dataset.control_nodes):
            group_order = np.random.permutation(num_groups)
            self.datapoints_order[i] = []
            for j in group_order:
                self.datapoints_order[i].extend(
                    np.arange(i*self.dataset.data.shape[0]+j*self.dataset.variations, i*self.dataset.data.shape[0]+(j+1)*self.dataset.variations))
            self.datapoints_order[i] = torch.Tensor(
                self.datapoints_order[i]).cuda()
        self.count = {i: i*self.dataset.data.shape[0]
                      for i in range(dataset.control_nodes)}

    def criterion(self, dist, k=1):
        num_nodes = int(np.sqrt(dist.size(2)))
        graph_uncertainty = Categorical(
            logits=dist).entropy().view((num_nodes, num_nodes))
        uncertain_nodes = torch.topk(
            graph_uncertainty.sum(0).abs().squeeze()[:self.dataset.control_nodes], k=k)[1]
        return uncertain_nodes.tolist()

    def sample(self, control_nodes, batch_size, num_batches=1):
        idx = []
        for control_node in control_nodes:
            for b in range(num_batches):
                start = int(self.count[control_node] %
                            self.dataset.data.shape[0])
                idx.extend(
                    self.datapoints_order[control_node][start:start+batch_size])
                self.count[control_node] += batch_size
        return torch.stack(idx).cuda()


class RandomDatasetSampler(AbstractSampler):
    """
    Samples groups from an existing dataset, instead of generating data on the fly by a simulator.
    """

    def __init__(self, dataset):
        assert isinstance(
            dataset, ControlOneGraphDataset), "To use dataset sampler, the dataset must be controlled!"
        self.dataset = dataset
        num_groups = int(self.dataset.data.shape[0]/self.dataset.variations)
        self.datapoints_order = {}
        for i in range(dataset.control_nodes):
            group_order = np.random.permutation(num_groups)
            self.datapoints_order[i] = []
            for j in group_order:
                self.datapoints_order[i].extend(
                    np.arange(i*self.dataset.data.shape[0]+j*self.dataset.variations, i*self.dataset.data.shape[0]+(j+1)*self.dataset.variations))
            self.datapoints_order[i] = torch.Tensor(
                self.datapoints_order[i]).cuda()
        self.count = {i: i*self.dataset.data.shape[0]
                      for i in range(dataset.control_nodes)}

    def criterion(self, dist, k=1):
        num_nodes = self.dataset.control_nodes
        return np.random.randint(low=0, high=num_nodes, size=k)

    def sample(self, control_nodes, batch_size, num_batches=1):
        idx = []
        for control_node in control_nodes:
            for b in range(num_batches):
                start = int(self.count[control_node] %
                            self.dataset.data.shape[0])
                idx.extend(
                    self.datapoints_order[control_node][start:start+batch_size])
                self.count[control_node] += batch_size
        return torch.stack(idx).cuda()


class MaximalEntropySimulatorSampler(AbstractSampler):
    def __init__(self, simulator):
        self.simulator = simulator

    def criterion(self, dist):
        num_nodes = int(np.sqrt(dist.size(2)))
        graph_uncertainty = Categorical(
            logits=dist).entropy().view((num_nodes, num_nodes))
        control_node = graph_uncertainty.sum(0).abs().argmax(-1)
        return control_node.item()

    def sample(self, control_node, batch_size):
        """
        Choose the node that causes maximal average uncertainty to other nodes
        dist: size (num_nodes**2, num_cls), assuming self-loops. The ith node's influence is on
        the ith column of the adjacency matrix, so the indices are [i:num_nodes**2::num_nodes]
        """
        inputs, targets = self.simulator.simulate(control_node, batch_size)
        control_node = torch.LongTensor([control_node]).expand(batch_size, 1)
        return self.simulator.merge_inputs_targets_onehot(inputs, targets), control_node


class RandomSimulatorSampler(AbstractSampler):
    # TODO: not tested
    def __init__(self, simulator):
        self.simulator = simulator

    def criterion(self, dist):
        num_nodes = int(np.sqrt(dist.size(2)))
        return np.random.randint(low=0, high=num_nodes)

    def sample(self, control_node, batch_size):
        """
        Choose the node that causes maximal average uncertainty to other nodes
        dist: size (num_nodes**2, num_cls), assuming self-loops. The ith node's influence is on
        the ith column of the adjacency matrix, so the indices are [i:num_nodes**2::num_nodes]
        """
        inputs, targets = self.simulator.simulate(control_node, batch_size)
        control_node = torch.LongTensor([control_node]).expand(batch_size, 1)
        return self.simulator.merge_inputs_targets_onehot(inputs, targets), control_node
