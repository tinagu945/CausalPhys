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


class AbstractDatasetSampler(AbstractSampler):
    def __init__(self, dataset, args):
        assert isinstance(
            dataset, ControlOneGraphDataset), "To use dataset sampler, the dataset must be controlled!"
        self.dataset = dataset
        if args.need_grouping:
            pass
            # # TODO: this branch hasn't been debugged.
            # # Number of groups for one node
            # num_groups = int(
            #     self.dataset.data.shape[0]/self.dataset.variations)
            # self.datapoints_order = {}
            # for i in range(dataset.control_nodes):
            #     group_order = np.random.permutation(num_groups)
            #     self.datapoints_order[i] = []
            #     for j in group_order:
            #         self.datapoints_order[i].extend(
            #             np.arange(i*self.dataset.data.shape[0]+j*self.dataset.variations, i*self.dataset.data.shape[0]+(j+1)*self.dataset.variations))
            #     self.datapoints_order[i] = torch.Tensor(
            #         self.datapoints_order[i]).cuda()
            # self.count = {i: i*self.dataset.data.shape[0]
            #               for i in range(dataset.control_nodes)}
        else:
            # Number of groups for one node
            self.num_groups_per_node = int(
                self.dataset.data.shape[0]/(self.dataset.control_nodes*self.dataset.variations))
            self.datapoints_order = {}
            group_starts = np.linspace(
                0, self.dataset.data.shape[0], self.dataset.control_nodes, endpoint=False)
            for i in range(dataset.control_nodes):
                group_order = np.random.permutation(self.num_groups_per_node)
                self.datapoints_order[i] = []
                for j in group_order:
                    idx_of_a_group = np.arange(
                        group_starts[i]+j*self.dataset.variations, group_starts[i]+(j+1)*self.dataset.variations, dtype=int)
                    self.datapoints_order[i].extend(idx_of_a_group)
            # Counting a node's number of datapoints queried so far.
            self.count = {i: 0 for i in range(dataset.control_nodes)}
            self.size_per_node = int(
                self.dataset.data.shape[0]/self.dataset.control_nodes)

    def criterion(self, dist):
        return NotImplementedError()

    def sample(self, control_nodes, group_size, num_groups):
        idx = []
        for control_node in control_nodes:
            for b in range(num_groups):
                start = self.count[control_node]
                idx.extend(
                    self.datapoints_order[control_node][start:start+group_size])
                self.count[control_node] += group_size
        return torch.LongTensor(idx).cuda()


class MaximalEntropyDatasetSampler(AbstractDatasetSampler):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)

    def criterion(self, graph, k=1):
        """Sort all nodes in the graph based on uncertainty sum of all outcoming edges, in decreasing order.
        """
        num_nodes = int(np.sqrt(graph.size(2)))
        graph_uncertainty = Categorical(
            logits=graph).entropy().view((num_nodes, num_nodes))
        uncertainty_sum = graph_uncertainty.sum(0).abs().squeeze()
        order = torch.argsort(-1 * uncertainty_sum)
        # Skip those nodes with no data left for query, or are the target variables.
        available_nodes = []
        for i in order.tolist():
            if i < self.dataset.control_nodes and self.count[i] < self.size_per_node:
                available_nodes.append(int(i))
        # import pdb
        # pdb.set_trace()
        print(uncertainty_sum)
        return available_nodes[:k]


class RandomDatasetSampler(AbstractDatasetSampler):
    """
    Samples groups from an existing dataset, instead of generating data on the fly by a simulator.
    """

    def __init__(self, dataset, args):
        super().__init__(dataset, args)

    def criterion(self, dist, k=1):
        num_nodes = self.dataset.control_nodes
        return np.random.randint(low=0, high=num_nodes, size=k).tolist()


# class MaximalEntropySimulatorSampler(AbstractSampler):
#     def __init__(self, simulator):
#         self.simulator = simulator

#     def criterion(self, dist):
#         num_nodes = int(np.sqrt(dist.size(2)))
#         graph_uncertainty = Categorical(
#             logits=dist).entropy().view((num_nodes, num_nodes))
#         control_node = graph_uncertainty.sum(0).abs().argmax(-1)
#         return control_node.item()

#     def sample(self, control_node, batch_size):
#         """
#         Choose the node that causes maximal average uncertainty to other nodes
#         dist: size (num_nodes**2, num_cls), assuming self-loops. The ith node's influence is on
#         the ith column of the adjacency matrix, so the indices are [i:num_nodes**2::num_nodes]
#         """
#         inputs, targets = self.simulator.simulate(control_node, batch_size)
#         control_node = torch.LongTensor([control_node]).expand(batch_size, 1)
#         return self.simulator.merge_inputs_targets_onehot(inputs, targets), control_node


# class RandomSimulatorSampler(AbstractSampler):
#     # TODO: not tested
#     def __init__(self, simulator):
#         self.simulator = simulator

#     def criterion(self, dist):
#         num_nodes = int(np.sqrt(dist.size(2)))
#         return np.random.randint(low=0, high=num_nodes)

#     def sample(self, control_node, batch_size):
#         """
#         Choose the node that causes maximal average uncertainty to other nodes
#         dist: size (num_nodes**2, num_cls), assuming self-loops. The ith node's influence is on
#         the ith column of the adjacency matrix, so the indices are [i:num_nodes**2::num_nodes]
#         """
#         inputs, targets = self.simulator.simulate(control_node, batch_size)
#         control_node = torch.LongTensor([control_node]).expand(batch_size, 1)
#         return self.simulator.merge_inputs_targets_onehot(inputs, targets), control_node
