import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

from torch.autograd import Variable
from utils.functions import my_softmax, get_offdiag_indices, gumbel_softmax
import GPUtil

_EPS = 1e-10


class MLPDecoder_Causal(nn.Module):
    """MLP decoder module."""

    def __init__(self, args, rel_rec, rel_send):
        super(MLPDecoder_Causal, self).__init__()
        self.rel_rec = rel_rec
        self.rel_send = rel_send
        self.tau = args.temp
        self.hard = args.hard
        self.pred_steps = args.prediction_steps
        self.msg_hook = []

        # TODO: only the last col of the original adj matrix will be trained
        if args.self_loop:
            self.rel_graph_shape = (1, 1, args.num_atoms**2, args.edge_types)
        else:
            self.rel_graph_shape = (
                1, 1, args.num_atoms*(args.num_atoms-1), args.edge_types)

        if args.gt_A:
            if args.suffix:
                edge = np.load(
                    'data/datasets/edges_train_causal_vel_' + args.suffix + '.npy')
            else:
                edge = np.load(
                    'data/datasets/edges_valid_causal_vel_' + args.val_suffix + '.npy')

            self.rel_graph = torch.from_numpy(edge*10).cuda()
            self.requires_grad = False
            print('Using ground truth A and the softmax result is', self.rel_graph)
        elif args.all_connect:
            self.rel_graph = torch.zeros(
                self.rel_graph_shape, requires_grad=False, device="cuda")
            self.rel_graph[:, :, :, 1] = 10.0
            print('Using fully connected adjacency matrix!', self.rel_graph)
        else:
            self.rel_graph = torch.zeros(
                self.rel_graph_shape, requires_grad=True, device="cuda")
            nn.init.xavier_normal_(self.rel_graph)

        self.edge_types = args.edge_types
        self.all_connect = args.all_connect

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * args.dims, args.decoder_hidden) for _ in range(args.edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(args.decoder_hidden, args.decoder_hidden) for _ in range(args.edge_types)])
        self.msg_out_shape = args.decoder_hidden
        self.skip_first_edge_type = args.skip_first
        # dim small to large to small
        self.out_fc1 = nn.Linear(
            args.dims + args.decoder_hidden, args.decoder_hidden)
        self.out_fc2 = nn.Linear(args.decoder_hidden, args.decoder_hidden)
        self.out_fc3 = nn.Linear(args.decoder_hidden, args.dims)

        self.dropout_prob = args.decoder_dropout

    def single_step_forward(self, single_timestep_inputs,
                            single_timestep_rel_type, msg_hook):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(self.rel_rec, single_timestep_inputs)
        senders = torch.matmul(self.rel_send, single_timestep_inputs)
        # senders size: [bs, 1, num_atoms*(num_atoms-1), num_dims]
        # pre_msg size: [bs, timesteps, num_atoms*(num_atoms-1), 2*num_dims]

        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                               pre_msg.size(2), self.msg_out_shape).cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # # Run separate MLP for every edge type
        # # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            # single_timestep_rel_type size: [bs, pred_steps, #node*(#node-1), #edge_type]
            # msg size: [bs, pred_steps, #node*(#node-1), self.msg_out_shape=256]
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # msg_hook.append(msg.transpose(1, 0))
        # all_msgs[:,:,-8,:] = all_msgs[:,:,-8,:]*0
        # Edge2node, Aggregate all msgs to receiver
        # agg_msg size [bs, pred_steps, #node, self.msg_out_shape=256]
        agg_msgs = all_msgs.transpose(-2, -
                                      1).matmul(self.rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()
        # single_timestep_inputs size [bs, pred_steps, #node, args.dims=9]. So aug_inputs [bs, pred_steps, #node, 256+9]
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        # import pdb
        # pdb.set_trace()
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)
        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs):
        # NOTE: Assumes that we have the same graph across all samples.
        edges = gumbel_softmax(self.rel_graph, self.tau, self.hard)
        inputs = inputs.transpose(1, 2).contiguous()
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        time_steps = inputs.size(1)

        assert (self.pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps) 5
        last_pred = inputs[:, 0::self.pred_steps, :, :]

#         curr_rel_type = rel_type[:, 0::pred_steps, :, :]

        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        # Run n prediction steps
        for step in range(0, self.pred_steps):
            last_pred = self.single_step_forward(
                last_pred, edges, self.msg_hook)
            preds.append(last_pred)
        sizes = [preds[0].size(0), preds[0].size(1) * self.pred_steps,
                 preds[0].size(2), preds[0].size(3)]
        output = torch.zeros(sizes).cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):  # 10
            # 5 fixed points, 10 each sequence. preds[i] means the ith of each sequence.
            output[:, i::self.pred_steps, :, :] = preds[i]
        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        del last_pred, preds
        return pred_all.transpose(1, 2).contiguous(), \
            self.rel_graph.squeeze(1).expand(
                [inputs.size(0), self.rel_graph_shape[2], self.edge_types]), None  # torch.cat(self.msg_hook, axis=1).cuda()
