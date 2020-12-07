import torch
import numpy as np
from torch.distributions import Categorical
from data.simulator import RolloutSimulator
from data.generate_dataset import generate_dataset_discrete
from AL.AL_control_sampler import AbstractSampler


class AbstractDatasetampler(AbstractSampler):
    def __init__(self, dataset):
        assert isinstance(
            dataset, OneGraphDataset), "To use no-control dataset sampler, the dataset must be not controlled!"
        self.dataset = dataset

    def criterion(self, dist):
        return NotImplementedError()

    def sample(self, idx):
        return self.dataset[idx]

# TODO:
# class RandomDatasetSampler(AbstractDatasetSampler):
#     def __init__(self, dataset):
#         super().__init__(dataset)

#     def criterion(self, dist):
#         return np.random.randint(len(self.dataset))


# class MaximalEntropyDatasetSampler(AbstractDatasetSampler):


class AbstractSimulatorSampler(AbstractSampler):
    def __init__(self, simulator):
        assert isinstance(
            simulator, RolloutSimulator), "To use no-control simulator sampler, the simulator must be not controlled!"
        self.simulator = simulator

    def criterion(self, dist):
        return NotImplementedError()

    def sample(self, input_setting):
        if self.simulator.noise:
            _, trajectory = generate_dataset_discrete(
                [setting_value], self.simulator.scenario, False)
        else:
            trajectory, _ = generate_dataset_discrete(
                [setting_value], self.simulator.scenario, False)

        trajectory = torch.Tensor(trajectory)

# TODO:
# class MaximalEntropySimulatorSampler(AbstractSimulatorSampler):
#     def __init__(self, simulator):
#         self.simulator = simulator

#     def criterion(self, dist):


class RandomSimulatorSampler(AbstractSimulatorSampler):
    def __init__(self, simulator, discrete_mapping):
        super().__init__(simulator)
        self.discrete_mapping = discrete_mapping

    def criterion(self):
        choice = []
        for m in self.discrete_mapping:
            choice.append(np.random.randint(len(self.discrete_mapping)))
        return choice
