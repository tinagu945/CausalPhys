import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from torch.autograd import Variable
from utils.functions import my_softmax, get_offdiag_indices, gumbel_softmax

_EPS = 1e-10


class MLPDecoder_Causal(nn.Module):
    """MLP decoder module."""

    def __init__(self, args):
        super(MLPDecoder_Causal, self).__init__()
        n_in_node = args.dims
        edge_types = args.edge_types
        msg_hid = args.decoder_hidden
        msg_out = args.decoder_hidden
        n_hid = args.decoder_hidden
        skip_first = args.skip_first
        num_nodes = args.num_atoms
        do_prob = args.decoder_dropout

        # TODO: only the last col of the original adj matrix will be trained
        if args.self_loop:
            self.rel_graph_shape = (1, 1, num_nodes**2, edge_types)
        else:
            self.rel_graph_shape = (1, 1, num_nodes*(num_nodes-1), edge_types)

        self.rel_graph = torch.zeros(
            self.rel_graph_shape, requires_grad=True, device="cuda")

        nn.init.xavier_normal_(self.rel_graph)

        self.num_nodes = num_nodes
        self.edge_types = edge_types

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first
        # dim small to large to small
        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type, msg_hook):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        # pre_msg size: [bs,timesteps, num_atoms*(num_atoms-1), 2]

        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))

        all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            # single_timestep_rel_type size: [bs, pred_steps, #node*(#node-1), #edge_type]
            # msg size: [bs, pred_steps, #node*(#node-1), self.msg_out_shape=256]
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        msg_hook.append(msg.transpose(1, 0))
        # all_msgs[:,:,-8,:] = all_msgs[:,:,-8,:]*0
        # Aggregate all msgs to receiver
        # agg_msg bs, pred_steps, #node, self.msg_out_shape=256
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        # msg bs, pred_steps, #node, self.msg_out_shape+1=257
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        # in:256, out:1
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_rec, rel_send, tau, hard, pred_steps, msg_hook):
        # NOTE: Assumes that we have the same graph across all samples.
        edges = gumbel_softmax(self.rel_graph, tau, hard)

        inputs = inputs.transpose(1, 2).contiguous()
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps) 5
        last_pred = inputs[:, 0::pred_steps, :, :]

#         curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(
                last_pred, rel_rec, rel_send, edges, msg_hook)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = torch.zeros(sizes).cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):  # 10
            # 5 fixed points, 10 each sequence. preds[i] means the ith of each sequence.
            output[:, i::pred_steps, :, :] = preds[i]
        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous(), \
            self.rel_graph.squeeze(1).expand(
                [inputs.size(0), self.rel_graph_shape[2], self.edge_types]), torch.cat(msg_hook).cuda()
