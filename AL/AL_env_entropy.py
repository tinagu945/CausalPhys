import time
import itertools
import os
import datetime
import sys
import numpy as np
import ast

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.distributions import Categorical

from models.modules_causal_vel import *
from models.RL_feature_extractors import *
from data.generate_dataset import merge_inputs_targets_onehot, generate_dataset_discrete
from train import train_control
from val import val_control
from data.datasets import RLDataset
from utils.early_stop import EarlyStopping

import gc
import GPUtil


class AL_env_entropy(object):
    """
    Wrap the training and testing of the causal network into an env for RL.
    """

    def __init__(self, args, rel_rec, rel_send, learning_assess_data, simulator, log_prior, logger, save_folder, valid_data_loader, edge, mins, maxs, discrete_mapping=None, discrete_mapping_grad=None, lstm_direction=2, feature_extractors=True):
        self.discrete_mapping_grad = discrete_mapping_grad
        self.args = args
        self.edge = edge
        self.mins = mins
        self.maxs = maxs
        self.epoch = -1
        self.learning_assess_data = learning_assess_data

        self.rel_rec = rel_rec
        self.rel_send = rel_send
        self.log_prior = log_prior
        self.lstm_direction = lstm_direction

        self.deepset = DeepSet(mode='sum').cuda()
        self.simulator = simulator
        self.valid_data_loader = valid_data_loader
        self.logger = logger
        self.save_folder = save_folder
        self.feature_extractors = feature_extractors

        #only useful if self.obj_num changes
        self.obj_num = self.args.initial_obj_num
        # list[i] contains all value choices for the ith attribute. #choices can be different across objects. Having this implies the setting is discrete, else None implies continuous.
        self.discrete_mapping = discrete_mapping
        # print(self.discrete_mapping)
        self.f = open(os.path.join(
            self.logger.save_folder, 'queries.txt'), 'a')
        
        values_ind=[[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]]
        all_values_ind = [list(i) for i in list(itertools.product(*values_ind))]
        self.ind_dict={str(all_values_ind[i]):i for i in range(len(all_values_ind))}
        self.ind_dict_rev={i:all_values_ind[i] for i in range(len(all_values_ind))} 
#         self.fff=[]
#         self.fff_count=0
#         with open('logs_RL/expAL_2020-12-29T01:23:39.058334_val-suffix_sliding_spaced_fixedgaussian0.1_new_interpolation_noise_0.1_val-size_1000_patience_10_budget_46656_gt-A/queries.txt', 'r') as f:
#             for l in f:
#                 print(l)
#                 try:
                    
#                     self.fff.append(ast.literal_eval(l))
#                 except:
#                     pass
#         print(len(self.fff))

    def init_train_data(self, data_num_per_obj=1):
        if self.discrete_mapping:
            self.init_data_queries = np.load(os.path.join(
            self.args.save_folder, 'init_data.npy'))
            print('env init_data', self.init_data_queries)
        
            for query in self.init_data_queries:
                complete_action_0=query.tolist()
                complete_action=[self.ind_dict[str(complete_action_0[:3])]]+complete_action_0[3:]
                print('env init_data converted', complete_action)
                new_datapoint, query_setting, _ = self.action_to_new_data(complete_action)
                # Doing intervention for overlapping objects ease the graph inference alot.
                repeat = self.process_new_data(
                    query_setting, new_datapoint, self.args.intervene)
                # print('initialize data with repeat', repeat)
                self.obj_data[int(complete_action[0])].append(new_datapoint)
                self.train_dataset.update(new_datapoint.clone())

    def action_to_new_data(self, action, idx_grad=None, action_grad=None):
        """
        Given the action encoded with RL agent's notation, get the datapoints it represents. Also put the new data into the training pool.

        Args:
            action ([type]): the obj index and the new trajectory setting action encoded with RL agent's notation.

        Returns: query setting includes the entire info of the obj.
        """
        assert len(action)==len(self.discrete_mapping)
        if self.discrete_mapping:
            # then the queries except the last propensity score are not actual values but index.
            setting_value = []
            setting_value_grad = []
            for i in range(len(self.discrete_mapping)):
                try:
                    setting_value.extend(
                        self.discrete_mapping[i][int(action[i])])
                except:
                    setting_value.append(
                        self.discrete_mapping[i][int(action[i])])

        else:
            setting_value = list(query)

        # Assume the cluster of expensive, obj-related variables are before the cluster of cheap, RL-chosen variables.
        if self.args.noise:
            _, trajectory = generate_dataset_discrete(
                [setting_value], self.simulator.scenario, False)
        else:
            trajectory, _ = generate_dataset_discrete(
                [setting_value], self.simulator.scenario, False)

        trajectory = torch.Tensor(trajectory)
        return trajectory, setting_value, None

    def process_new_data(self, action, new_datapoint, intervene):
        """Do intervention on the rel_graph while check repeatance of the action

        Args:
            action ([type]): [description] contains gradient
            new_datapoint ([type]): [description] rollout of the action, no gradient.
        """
        repeat = 0
        num_intervention=0
        # new datapoint: (1, #num nodes, #steps, #dim)
        if torch.is_tensor(action) and action.requires_grad:
            m = nn.ConstantPad1d((0, 2), 0)
            new_setting = action
            for l in self.obj_data.values():
                if l:
                    for d in l:
                        if intervene:
                            setting = d[0, :-2, 0, 0].cuda()
                            causal = abs(new_setting-setting).unsqueeze(0)
                            caused = abs(
                                new_datapoint[0, :, -1, 0]-d[0, :, -1, 0]).unsqueeze(1).cuda()
                            relations = (m(causal)*caused).flatten()
                            relations_sum = relations.sum()
                            coeff = 1e-2/relations_sum

                            relations = torch.cat(
                                [-coeff*relations.unsqueeze(-1), coeff*relations.unsqueeze(-1)], axis=1)

                            self.causal_model.rel_graph = self.causal_model.rel_graph.clone() + \
                                relations.unsqueeze(0).unsqueeze(0)

        else:
            with torch.no_grad():
                # Threshold is scaling of noise variance.
                threshold = self.args.causal_threshold * \
                    self.simulator.scenario.add_noise['value']
                threshold = torch.from_numpy(np.concatenate(
                    [np.zeros((self.args.input_atoms, self.args.timesteps)), threshold]))
                new_setting = new_datapoint[0, :, 0, 0]
                for l in self.obj_data.values():
                    for d in l:
#                         if l:
#                         d = l[0]
                        setting = d[0, :, 0, 0]
                        no_overlap = (abs(new_setting-setting)
                                      > threshold[:, 0]).sum()
                        if intervene and no_overlap == 1:
                            self.total_intervention += 1
                            num_intervention += 1
                            # found a single perfect intervention.
                            a = torch.abs(new_setting-setting)
                            b = torch.abs(
                                new_datapoint[0, :, -1, 0]-d[0, :, -1, 0])
                            # causal is a single number, but caused can be a list.
                            causal = torch.nonzero(a > threshold[:, 0]).item()
                            if causal not in self.intervened_nodes:
                                self.intervened_nodes.append(causal)
                            caused = torch.nonzero(
                                b > threshold[:, -1]).flatten().numpy()
                            # Avoid counting the causal node in the caused list due to its own change at first.
                            if a[causal] == b[causal]:
                                caused = caused[caused != causal]
                            # With noise, we can't be certain even when some nodes change with some intervention above some threshold, so we only add the corresponding probability.
                            if len(caused) > 0:
                                self.causal_model.rel_graph[0,
                                                            0, caused*self.args.num_atoms+causal, 1] += self.args.intervene_strength
                            no_caused = np.array(list(
                                set(np.arange(self.args.num_atoms))-set(caused)))
                            self.causal_model.rel_graph[0,
                                                        0, no_caused*self.args.num_atoms+causal, 0] += self.args.intervene_strength

                        if no_overlap.item() == 0:
                            repeat += 1

                            # causal = abs(
                            #     new_setting-setting).unsqueeze(0)
                            # caused = abs(
                            #     new_datapoint[0, :, -1, 0]-d[0, :, -1, 0]).unsqueeze(1).cuda()
                            # relations = (causal*caused).flatten()

                            # self.causal_model.rel_graph[0, 0, :,
                            #                             0] -= relations*5
                            # self.causal_model.rel_graph[0, 0, :,
                            # 1] += relations*5
        return repeat, num_intervention

    def train_causal(self):
        """[summary]

            Args:
                action ([list]): the new trajectory setting.

            Returns:
                state: [learning_assess, obj_data_features]
                reward: - val_MSE
                done: Whether the data budget is met. If yes, the training can end early.
            """
        # GPUtil.showUtilization()
        # self.train_dataset.data size (batch_size, num_nodes, timesteps, feat_dims)
#         print('dataset size', len(self.train_dataset),
#               'last ten', self.train_dataset.data[-10:, :, 0, 0])
        train_data_loader = DataLoader(
            self.train_dataset, batch_size=self.args.train_bs, shuffle=False)
        for i in range(self.args.max_causal_epochs):
            print(str(i), "iter of epoch", self.epoch)
            nll, nll_lasttwo, kl, mse, control_constraint_loss, lr, rel_graphs, rel_graphs_grad, a, b, c, d, e = train_control(
                self.args, self.log_prior, self.causal_model_optimizer, self.save_folder, train_data_loader, self.causal_model, self.epoch)

            # val_dataset should be continuous, more coverage
            nll_val, nll_lasttwo_val, kl_val, mse_val, a_val, b_val, c_val, control_constraint_loss_val, nll_lasttwo_5_val, nll_lasttwo_10_val, nll_lasttwo__1_val, nll_lasttwo_1_val = val_control(
                self.args, self.log_prior, self.logger, self.save_folder, self.valid_data_loader, self.epoch, self.causal_model)
            # val_loss = 0
            self.scheduler.step()
            self.early_stop_monitor(nll_val)
            if self.early_stop_monitor.counter == self.args.patience:
                self.early_stop_monitor.counter = 0
                self.early_stop_monitor.best_score = None
                self.early_stop_monitor.stopped_epoch = i
                print("Early stopping", str(i), "iter of epoch", self.epoch)
                break

        self.logger.log('val', self.causal_model, self.epoch, nll_val, nll_lasttwo_val, kl_val=kl_val, mse_val=mse_val, a_val=a_val, b_val=b_val, c_val=c_val, control_constraint_loss_val=control_constraint_loss_val,
                        nll_lasttwo_5_val=nll_lasttwo_5_val,  nll_lasttwo_10_val=nll_lasttwo_10_val, nll_lasttwo__1_val=nll_lasttwo__1_val, nll_lasttwo_1_val=nll_lasttwo_1_val, scheduler=self.scheduler)

        if self.epoch % self.args.train_log_freq == 0:
            self.logger.log('train', self.causal_model, self.epoch, nll, nll_lasttwo, kl_train=kl, mse_train=mse, control_constraint_loss_train=control_constraint_loss, lr_train=lr, rel_graphs=rel_graphs,
                            rel_graphs_grad=rel_graphs_grad, msg_hook_weights_train=a, nll_lasttwo_5_train=b, nll_lasttwo_10_train=c, nll_lasttwo__1_train=d, nll_lasttwo_1_train=e)

        self.epoch += 1
        return nll_val
    
    def step_entropy(self, num_intervention):
        state=None
        if self.feature_extractors:
            state = self.extract_features()
        # Not punishing repetitions, only log it.
#         import pdb;pdb.set_trace()
#         penalty = Categorical(
#             logits=self.causal_model.rel_graph).entropy().sum() + 1
#         reward =10*self.total_intervention-1
        reward = num_intervention
        #self.train_dataset.data.size(0) + 100*repeat
        # used up budget or succeed before it
#         done = (penalty < self.args.budget) or (
#             self.train_dataset.data.size(0) > 150)
#         done = (reward > self.args.budget) or (
#             self.train_dataset.data.size(0) > 150)
        done = (self.train_dataset.data.size(0) == self.args.budget)

        return state, reward, done
    

    def extract_features(self):
        # batch_size, num_nodes, step, dim
        # TODO: change it to include one hot feature
#         import pdb;pdb.set_trace()
        pred = self.causal_model(self.learning_assess_data[:10,:,:,:])[0][:, :, :, 0]
        pred = pred.transpose(1, 2).detach()
        # learning_assess = self.learning_extractor(pred)
        learning_assess = pred
        learning_assess_feat = torch.zeros(
            (learning_assess.size(0), 2*self.args.extract_feat_dim)).cuda()
        for k in range(learning_assess.size(0)):
            out, [h, c] = self.learning_assess_extractor(
                learning_assess[k:k+1])
            unpack = out.view(-1, 1, self.lstm_direction,
                              self.args.extract_feat_dim)
            # (batch_size, hidden_size)
            learning_assess_feat[k:k+1] = torch.cat(
                (unpack[-1, :, 0, :], unpack[0, :, 1, :]), axis=1)

        # (1, 2 * feature_dims)
        set_learning_assess_feat = self.deepset(
            learning_assess_feat, axis=0, keepdim=False)

        obj_data_features = []
        for i in self.obj.keys():
            obj_param = torch.Tensor(self.obj[i]).cuda()
            # TODO: change it to include one hot feature
            # (batch, step, num_nodes), batch_first inputs
            if self.obj_data[i]:
                obj_traj = torch.cat(self.obj_data[i]).cuda()[
                    :, :, :, 0].transpose(1, 2)

                all_traj_feat = torch.zeros(
                    (obj_traj.size(0), 2*self.args.extract_feat_dim)).cuda()
                for j in range(obj_traj.size(0)):
                    # (seq_len, batch, num_directions * hidden_size)
                    out, [h, c] = self.obj_data_extractor(
                        obj_traj[j:j+1, :, :])
                    unpack = out.view(-1, 1, self.lstm_direction,
                                      self.args.extract_feat_dim)
                    # (batch_size, 2 * hidden_size)
                    all_traj_feat[j:j +
                                  1] = torch.cat((unpack[-1, :, 0, :], unpack[0, :, 1, :]), axis=1)
            else:
                all_traj_feat = torch.zeros(
                    (1, 2*self.args.extract_feat_dim)).cuda()

            set_traj_feat = self.deepset(all_traj_feat, axis=0, keepdim=True)
            # (1, 3 * feature_dims)
            obj_data_features.append(
                torch.cat([self.obj_extractor(obj_param).unsqueeze(0), set_traj_feat], axis=1))

        # obj_data_features = [torch.zeros((1, 3*self.args.extract_feat_dim)).cuda()]*6
        obj_data_features = torch.cat(obj_data_features).flatten()
        return torch.cat([set_learning_assess_feat, obj_data_features])

    
    def extract_features_simple(self):
        obj_data_features = []
        for i in self.obj.keys():
            obj_param = torch.Tensor(self.obj[i]).cuda()
            # TODO: change it to include one hot feature
            # (batch, step, num_nodes), batch_first inputs
            if self.obj_data[i]:
                obj_traj = torch.cat(self.obj_data[i]).cuda()[
                    :, :, 0, 0].transpose(1, 2)
                all_traj_feat = torch.zeros(
                    (obj_traj.size(0), 2*self.args.extract_feat_dim)).cuda()
                for j in range(obj_traj.size(0)):
                    feat = self.obj_data_extractor(obj_traj[j:j+1, :, :])
                    all_traj_feat[j:j + 1] = feat
                    
            set_traj_feat = self.deepset(all_traj_feat, axis=0, keepdim=True)
            obj_data_features.append(set_traj_feat)
        return [obj_data_features]

#     def extract_features_1(self):
#         set_learning_assess_feat = torch.zeros((1, 128)).cuda()
#         obj_data_features = [torch.zeros((1, 192)).cuda()]*6
#         return [set_learning_assess_feat, obj_data_features]


    def reset(self):  # data_num_per_obj=1):
        self.total_intervention = 0
        self.early_stop_monitor = EarlyStopping()
        # obj idx: obj attributes tensor
        self.obj = {i: self.discrete_mapping[0][i]
                    for i in range(self.args.initial_obj_num)}
        # obj idx: tensor datapoints using that obj. Acts as the training pool.
        self.obj_data = {i: [] for i in range(self.obj_num)}
        self.train_dataset = RLDataset(
            torch.Tensor(), self.edge, self.mins, self.maxs)
        self.intervened_nodes = []

        self.causal_model = MLPDecoder_Causal(
            self.args, self.rel_rec, self.rel_send).cuda()
        self.causal_model_optimizer = optim.Adam(list(self.causal_model.parameters())+[self.causal_model.rel_graph],
                                                 lr=self.args.lr)
        self.scheduler = lr_scheduler.StepLR(self.causal_model_optimizer, step_size=self.args.lr_decay,
                                             gamma=self.args.gamma)
        
##### For invertion with fixed initial datapoint testing only.
#         new_datapoint, query_setting, _ = self.action_to_new_data([0,3,0,5])
#         self.obj_data[0].append(new_datapoint)
#         self.train_dataset.update(new_datapoint.clone())
#####

#         self.init_train_data()
#         load_weights='100_warmup_weights.pt'
#         if load_weights not in os.listdir(self.args.save_folder):
#             print('no pretrained warm up weights, so training one now.')
#             train_data_loader = DataLoader(
#                 self.train_dataset, batch_size=self.args.train_bs, shuffle=False)

#             lowest_loss=np.inf
#             for i in range(1000):
#                 print(str(i), 'of warm up training', self.args.save_folder, lowest_loss)
#                 nll, nll_lasttwo, kl, mse, control_constraint_loss, lr, rel_graphs, rel_graphs_grad, a, b, c, d, e = train_control(
#                     self.args, self.log_prior, self.causal_model_optimizer, self.save_folder, train_data_loader, self.causal_model, self.epoch)

#                 # val_dataset should be continuous, more coverage
#                 nll_val, nll_lasttwo_val, kl_val, mse_val, a_val, b_val, c_val, control_constraint_loss_val, nll_lasttwo_5_val, nll_lasttwo_10_val, nll_lasttwo__1_val, nll_lasttwo_1_val = val_control(
#                     self.args, self.log_prior, self.logger, self.save_folder, self.valid_data_loader, self.epoch, self.causal_model)
#                 if nll_val<lowest_loss:
#                     print('new lowest_loss', nll_val)
#                     lowest_loss=nll_val
#                     torch.save([self.causal_model.state_dict(), self.causal_model.rel_graph],
#                        os.path.join(self.args.save_folder, load_weights))
                

#                 self.logger.log('val', self.causal_model, i, nll_val, nll_lasttwo_val, kl_val=kl_val, mse_val=mse_val, a_val=a_val, b_val=b_val, c_val=c_val, control_constraint_loss_val=control_constraint_loss_val,
#                                 nll_lasttwo_5_val=nll_lasttwo_5_val,  nll_lasttwo_10_val=nll_lasttwo_10_val, nll_lasttwo__1_val=nll_lasttwo__1_val, nll_lasttwo_1_val=nll_lasttwo_1_val, scheduler=self.scheduler)

#                 self.logger.log('train', self.causal_model, i, nll, nll_lasttwo, kl_train=kl, mse_train=mse, control_constraint_loss_train=control_constraint_loss, lr_train=lr, rel_graphs=rel_graphs,
#                                 rel_graphs_grad=rel_graphs_grad, msg_hook_weights_train=a, nll_lasttwo_5_train=b, nll_lasttwo_10_train=c, nll_lasttwo__1_train=d, nll_lasttwo_1_train=e)
                
#         else:
#             weights, graph = torch.load(os.path.join(
#                 self.args.save_folder, load_weights))
#             self.causal_model.load_state_dict(weights)
#             self.causal_model.rel_graph = graph.cuda()
#             print('warm up weights loaded.')
        
        if self.feature_extractors:
            # Make sure the output dim of both encoders are the same!
            self.obj_extractor = MLPEncoder(
                self.args, 3, 128, self.args.extract_feat_dim).cuda()
            self.obj_extractor_optimizer = optim.Adam(
                list(self.obj_extractor.model.parameters()), lr=self.args.obj_extractor_lr)
            # TODO: try self.obj_data_extractor = MLPEncoder(args, 3, 64, 16).cuda()
            # Bidirectional LSTM
            self.obj_data_extractor = LSTMEncoder(
                self.args.num_atoms, self.args.extract_feat_dim, num_direction=self.lstm_direction, batch_first=True).cuda()
            self.obj_data_extractor_optimizer = optim.Adam(
                list(self.obj_data_extractor.model.parameters())+[self.obj_data_extractor.h0, self.obj_data_extractor.c0], lr=self.args.obj_data_extractor_lr)
            self.learning_assess_extractor = LSTMEncoder(
                8, self.args.extract_feat_dim, num_direction=self.lstm_direction, batch_first=True).cuda()
            self.learning_assess_extractor_optimizer = optim.Adam(list(
                self.learning_assess_extractor.parameters())+[self.learning_assess_extractor.h0, self.learning_assess_extractor.c0], lr=self.args.learning_assess_extractor_lr)
            
#             checkpoint = torch.load('logs_RL/expRL_PPO_2021-01-27T13:01:29.595756_val-suffix_sliding_spaced_fixedgaussian0.1_new_interpolation_noise_0.1_causal-threshold_0.1_intervene_val-size_1000_patience_1_rl-update-timestep_512_rl-epochs_50000_rl-lr_1e-3_budget_17_K-epochs_20/PPO_others_34816.pth')
            
#             count = 0
#             with torch.no_grad():
#                 for i in self.obj_extractor.parameters():
#                     i.copy_(checkpoint['obj_extractor'][count])
#                     count += 1
# #                 print('checkpoint[obj_extractor]', len(checkpoint['obj_extractor']), count)

#                 count = 0
#                 for i in self.obj_data_extractor.model.parameters():
#                     i.copy_(checkpoint['obj_data_extractor'][count])
#                     count += 1
#                 self.obj_data_extractor.h0.copy_(checkpoint['obj_data_extractor'][count])
#                 count += 1
#                 self.obj_data_extractor.c0.copy_(checkpoint['obj_data_extractor'][count])
#                 count += 1
# #                 print('checkpoint[obj_data_extractor]', len(checkpoint['obj_data_extractor']), count)

#                 count = 0
#                 for i in self.learning_assess_extractor.model.parameters():
#                     i.copy_(checkpoint['learning_assess_extractor'][count])
#                     count += 1
#                 self.learning_assess_extractor.h0.copy_(checkpoint['learning_assess_extractor'][count])
#                 count += 1
#                 self.learning_assess_extractor.c0.copy_(checkpoint['learning_assess_extractor'][count])
#                 count += 1
#                 print('checkpoint[learning_assess_extractor]', len(checkpoint['learning_assess_extractor']), count)
#                 print('All loaded!')

#         if self.feature_extractors:
#             # Make sure the output dim of both encoders are the same!
#             self.obj_extractor = MLPEncoder(
#                 self.args, 3, 128, self.args.extract_feat_dim).cuda()
#             self.obj_extractor_new = MLPEncoder(
#                 self.args, 3, 128, self.args.extract_feat_dim).cuda()
#             self.obj_extractor_new.load_state_dict(self.obj_extractor.state_dict())
#             self.obj_extractor_optimizer = optim.Adam(
#                 list(self.obj_extractor_new.parameters()), lr=self.args.obj_extractor_lr)

#             # TODO: try self.obj_data_extractor = MLPEncoder(args, 3, 64, 16).cuda()
#             # Bidirectional LSTM
#             self.obj_data_extractor = LSTMEncoder(
#                 self.args.num_atoms, self.args.extract_feat_dim, num_direction=self.lstm_direction, batch_first=True).cuda()
#             self.obj_data_extractor_new = LSTMEncoder(
#                 self.args.num_atoms, self.args.extract_feat_dim, num_direction=self.lstm_direction, batch_first=True).cuda()
#             self.obj_data_extractor_new.load_state_dict(self.obj_data_extractor.state_dict())
#             self.obj_data_extractor_optimizer = optim.Adam(
#                 list(self.obj_data_extractor_new.parameters()), lr=self.args.obj_data_extractor_lr)

#             self.learning_assess_extractor = LSTMEncoder(
#                 8, self.args.extract_feat_dim, num_direction=self.lstm_direction, batch_first=True).cuda()
#             self.learning_assess_extractor_new = LSTMEncoder(
#                 8, self.args.extract_feat_dim, num_direction=self.lstm_direction, batch_first=True).cuda()
#             self.learning_assess_extractor_new.load_state_dict(self.learning_assess_extractor.state_dict())
#             self.learning_assess_extractor_optimizer = optim.Adam(list(
#                 self.learning_assess_extractor_new.parameters()), lr=self.args.learning_assess_extractor_lr)
