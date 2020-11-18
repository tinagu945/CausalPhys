import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from torch.autograd import Variable
from utils import my_softmax, get_offdiag_indices, gumbel_softmax

_EPS = 1e-10


class MLPEncoder_Causal(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLPEncoder_Causal, self).__init__()

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    
    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        return self.fc_out(x)

    
    
    
class MLPDecoder_Causal(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False, cuda=True, num_nodes=7, pred_steps=1):
        super(MLPDecoder_Causal, self).__init__()
        
        #TODO: only the last col of the original adj matrix will be trained   
        if cuda:
            self.rel_graph = torch.zeros((1, 1, num_nodes*(num_nodes-1), edge_types), requires_grad=True, device="cuda")
#             self.rel_graph = torch.zeros((1, 1, num_nodes-1, edge_types), requires_grad=True, device="cuda")
#             self.zeros = torch.zeros((1, 1, (num_nodes-1)**2, edge_types), requires_grad=False, device="cuda")
#             self.all = torch.cat((self.zeros,self.rel_graph),dim=2)
        else:
            self.rel_graph = torch.zeros((1, 1, num_nodes*(num_nodes-1), edge_types), requires_grad=True)
#         import pdb;pdb.set_trace()
        
        nn.init.xavier_normal_(self.rel_graph)
        
        self.num_nodes = num_nodes
        self.edge_types = edge_types
#         for i in range(0,num_nodes*(num_nodes-1)+1, num_nodes-1):
#             self.rel_graph[:,:,i,:]=random.random()
   
            
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first
        #dim small to large to small
        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send, 
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs) 
        senders = torch.matmul(rel_send, single_timestep_inputs)
        #pre_msg bs,timesteps, num_atoms*(num_atoms-1), 2

        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
            

#         test=torch.zeros((1,1,56,1)).cuda()  51,52,54,55
#         test[:,:,20,:]=1
#         test[:,:,27,:]=1
#         test[:,:,41,:]=1
#         test[:,:,48,:]=1

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
#             print('start', i)
            
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            #single_timestep_rel_type bs, pred_steps, #node*(#node-1), #edge_type
            #msg bs, pred_steps, #node*(#node-1), self.msg_out_shape=256
#             msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]

            msg = msg * self.rel_graph[:, :, :, i:i + 1]
            all_msgs += msg
            

        # Aggregate all msgs to receiver
        #agg_msg bs, pred_steps, #node, self.msg_out_shape=256
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        #msg bs, pred_steps, #node, self.msg_out_shape+1=257
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        #in:256, out:1
        pred = self.out_fc3(pred)
#         import pdb; pdb.set_trace()

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_rec, rel_send, tau, hard, pred_steps):
        # NOTE: Assumes that we have the same graph across all samples.
#         logits = self.rel_graph # (inputs, rel_rec, rel_send)
        edges = gumbel_softmax(self.rel_graph, tau, hard)
        
        inputs = inputs.transpose(1, 2).contiguous()

#         sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
#                  rel_type.size(2)]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
#         rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps) 5
        last_pred = inputs[:, 0::pred_steps, :, :]

#         curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps): 
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send, edges)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)): #10
            #5 fixed points, 10 each sequence. preds[i] means the ith of each sequence.
            output[:, i::pred_steps, :, :] = preds[i] 
        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous(), \
    self.rel_graph.squeeze(1).expand([inputs.size(0), self.num_nodes*(self.num_nodes-1), self.edge_types])