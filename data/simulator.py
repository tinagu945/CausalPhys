import torch
import torch.nn as nn
import numpy as np


class AbstractSimulator(object):
    def __init__(self, scenario):
        """
        func: the physics equation used to simulate target vars by input vars
        """
        self.scenario = scenario

    def simulate(self, inputs):
        raise NotImplementedError()


class ControlSimulator(AbstractSimulator):
    """
    Used for AL
    """

    def __init__(self, scenario, lows, highs):
        """
        low: list of length #input_nodes. The lowest value an input var can take, ordered as input nodes.
        high: list of length #input_nodes. The highest value an input var can take, ordered as input nodes.
        """

        super().__init__(scenario)
        self.lows = lows
        self.highs = highs

    def simulate(self, control_idx, batch_size):
        inputs = []
        for i in range(self.scenario.num_inputs):
            inputs.append(
                np.random.uniform(low=self.lows[i], high=self.highs[i]))
        inputs = np.repeat(np.expand_dims(inputs, 0), batch_size, axis=0)
        # Input vars are constant over the trajectory, and same for all datapoints in one batch
        # Controlled var is constant over the trajectory but different for all datapoints in one batch
        control = np.random.uniform(
            low=self.lows[control_idx], high=self.highs[control_idx], size=(batch_size, 1))
        inputs[:, control_idx:control_idx+1] = control
        inputs, targets = self.scenario.rollout_func(inputs)
        return inputs, targets
