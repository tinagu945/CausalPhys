import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, args, state_dim, hidden_dim, out_dim):
        super(MLPEncoder, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim*2, out_dim),
        )

    def forward(self, x):
        return self.model(x)


class LSTMEncoder(nn.Module):
    """LSTM decoder module."""

    def __init__(self, input_size, hidden_size, num_layers=1, num_direction=2, **kwargs):
        super(LSTMEncoder, self).__init__()
        self.model = nn.LSTM(input_size, hidden_size,
                             num_layers, bidirectional=(num_direction == 2), **kwargs)
        self.h0 = torch.randn(num_layers*num_direction, 1, hidden_size).cuda()
        self.c0 = torch.randn(num_layers*num_direction, 1, hidden_size).cuda()

    def forward(self, x):
        # input has shape (seq_len, batch, input_size).
        # output has shape (seq_len, batch, num_layers * hidden_size)
        # hn and cn have shape (num_layers, batch, hidden_size)
        return self.model(x, (self.h0, self.c0))


class DeepSet(nn.Module):
    def __init__(self, mode='sum'):
        super(DeepSet, self).__init__()
        self.mode = mode

    def forward(self, x, axis=1, keepdim=True):
        # TODO: maybe try max pooling in the future.
        if self.mode == 'sum':
            return x.sum(axis, keepdim=keepdim)
