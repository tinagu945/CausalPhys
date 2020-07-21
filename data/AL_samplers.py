import torch
from torch.distributions import Categorical
import numpy as np


class MaximalEntropyDatasetSampler():
    # TODO: not tested
    def __init__(self, dataset):
        """Sample from a dataset like ControlDataset"""
        self.count={i:0 for i in range(control_nodes)}
        self.dataset= dataset
    
    def sample(self, dist, bastch_size):        
        uncertain_node = Categorical(probs = dist).entropy().argmax(-1)
        
        count = self.count[uncertain_node]
        base = len(self.dataset)*uncertain_node
        idx = base+count*self.dataset.variations
        
        self.count[uncertain_node] += 1
        #return data indices and repeated node idx
        return np.range(idx, idx+self.dataset.variations), uncertain_node.tile((1,1,self.dataset.variations,1))
    
    
    
class MaximalEntropySimulatorSampler():
    def __init__(self, simulator):
        self.simulator = simulator
        
    def criterion(self, dist):
        num_nodes = int(np.sqrt(dist.size(0)))
        graph_uncertainty = Categorical(probs = dist).entropy().resize((num_nodes, num_nodes))
        control_node = graph_uncertainty.sum(1).argmax(-1)
        return control_node.item()        
    
    def sample(self, control_node, batch_size):
        """ 
        Choose the node that causes maximal average uncertainty to other nodes
        dist: size (num_nodes**2, num_cls), assuming self-loops. The ith node's influence is on 
        the ith column of the adjacency matrix, so the indices are [i:num_nodes**2::num_nodes]  
        """      
        inputs, targets = self.simulator.simulate(control_node, batch_size)
        return self.simulator.merge_inputs_targets_onehot(inputs, targets), control_node.expand(1,1,self.batch_size,1)

    
    
class RandomSimulatorSampler():
    def __init__(self, simulator):
        self.simulator = simulator
    
    def sample(self, num_nodes, batch_size):        
        control_node = np.random.randint(low=0, high=num_nodes)
        return self.simulator.simulate(control_node, batch_size), control_node.tile((1,1,self.batch_size,1))