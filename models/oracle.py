import torch
import torch.nn as nn
import numpy as np


class AbstractOracle(object):
    def __init__(self, func):
        """
        func: the physics equation used to simulate target vars by input vars
        """
        self.func = func

        
    def simulate(self, inputs):
        raise NotImplementedError()
        
        
        
class ControlOracle(AbstractOracle):
    """
    Used for AL
    """
    def __init__(self, func, trajectory_len, num_nodes, low=0, high=100, \
                 control_low=20, control_high=50):
        """
        low: the lowest value an input var can take
        high: the highest value an input var can take
        control_low: the lowest value the controlled input var can take
        """
        
        super().__init__(func)
        self.trajectory_len = trajectory_len
        self.num_nodes = num_nodes
        self.control_low = control_low
        self.control_high= control_high
        
        
    def simulate(self, control_idx, batch_size):
        inputs = torch.FloatTensor(1, self.num_nodes, 1, 1).uniform_(self.low, self.high)
        # Input vars are constant over the trajectory, and same for all datapoints in one batch
        # Controlled var is constant over the trajectory but different for all datapoints in one batch
        inputs = inputs.repeat(self.batch_size, self.num_nodes, self.trajectory_len, 1)
        
        control = torch.FloatTensor(batch_size, , 1, 1).uniform_(self.low, self.high)
        control = inputs.repeat(batch_size, 1, self.trajectory_len, 1)
        
        inputs[:, control_idx, :, :] = control
        targets = self.func(inputs)
        return inputs, targets
        
    def merge_inputs_targets_onehot(self, inputs, targets)
        data = torch.cat((inputs, targets), dim=-1)        
        outputs = torch.zeros((data.size(0), data.size(1), data.size(2), 1+data.size(1)))
        outputs[:,:,:,0]=data                    
        # Add one hot encoding
        for i in range(1, outputs.size(-1)):
            outputs[:,:,:,i]=1         
        return outputs
        
