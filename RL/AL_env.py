import time
import argparse
import os
import datetime
import sys
import numpy as np

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn

from models.modules_causal_vel import *
from models.RL_feature_extractors import *
from data.generate_dataset import merge_inputs_targets_onehot
from train import train_control
from val import val_control
from data.datasets import RLDataset

import gc
import GPUtil


class AL_env(object):
    """
    Wrap the training and testing of the causal network into an env for RL.
    """

    def __init__(self, args, decoder, optimizer, scheduler, learning_assess_data, simulator, log_prior, logger, save_folder, valid_data_loader, edge, mins, maxs, discrete_mapping=None, discrete_mapping_grad=None, lstm_direction=2):
        self.discrete_mapping_grad = discrete_mapping_grad
        self.args = args
        self.edge = edge
        self.mins = mins
        self.maxs = maxs
        self.epoch = 0
        self.feature_dims = self.args.extract_feat_dim
        self.learning_assess_data = learning_assess_data
        self.causal_model = decoder
        self.causal_model_optimizer = optimizer
        self.log_prior = log_prior
        self.lstm_direction = lstm_direction
        # Make sure the output dim of both encoders are the same!
        self.obj_extractor = MLPEncoder(
            args, 3, 128, self.feature_dims).cuda()
        self.obj_extractor_optimizer = optim.Adam(
            list(self.obj_extractor.parameters()), lr=args.obj_extractor_lr)

        # TODO: try self.obj_data_extractor = MLPEncoder(args, 3, 64, 16).cuda()
        # Bidirectional LSTM
        self.obj_data_extractor = LSTMEncoder(
            args.num_atoms, self.feature_dims, num_direction=self.lstm_direction, batch_first=True).cuda()
        self.obj_data_extractor_optimizer = optim.Adam(
            list(self.obj_data_extractor.parameters()), lr=args.obj_data_extractor_lr)

        self.learning_assess_extractor = LSTMEncoder(
            8, self.feature_dims, num_direction=self.lstm_direction, batch_first=True).cuda()
        self.learning_assess_extractor_optimizer = optim.Adam(list(
            self.learning_assess_extractor.parameters()), lr=args.learning_assess_extractor_lr)

        # Change the shape of prediction to match with obj and data features.
        # self.learning_extractor = MLPEncoder(
        #     args, args.num_atoms, self.feature_dims, self.feature_dims).cuda()

        self.deepset = DeepSet(mode='sum').cuda()
        self.simulator = simulator
        self.valid_data_loader = valid_data_loader
        self.logger = logger
        self.save_folder = save_folder
        self.scheduler = scheduler

        self.obj_num = self.args.initial_obj_num
        # obj idx: obj attributes tensor
        self.obj = {}
        # obj idx: tensor datapoints using that obj. Acts as the training pool.
        self.obj_data = {i: [] for i in range(self.args.initial_obj_num)}
        # matrix of size #cheap_params x num_variations, how each action index maps to a setting value. Having this implies the setting is discrete, else None implies continuous.
        self.discrete_mapping = discrete_mapping

    def init_train_data(self, data_num_per_obj=1):
        # Only handles discrete case now
        if self.discrete_mapping:
            num_cheap_var = len(self.discrete_mapping)
            for i in range(self.obj_num):
                for j in range(data_num_per_obj):
                    setting = np.random.randint(
                        0, self.args.variations, size=num_cheap_var-1)
                    # The last is propensity score.
                    setting = np.append(setting, np.random.uniform())
                    new_datapoint, query_setting, _ = self.action_to_new_data([
                        i, setting])
                    new_datapoint = torch.Tensor(new_datapoint)
                    _ = self.process_new_data(
                        query_setting, new_datapoint, self.args.intervene)
                    self.obj_data[i].append(new_datapoint)

    def action_to_new_data(self, action, idx_grad=None, action_grad=None):
        """
        Given the action encoded with RL agent's notation, get the datapoints it represents.

        Args:
            action ([type]): the obj index and the new trajectory setting action encoded with RL agent's notation.

        Returns:
            list: the obj index and the queried trajectory.
        """
        idx, query = action
        if self.discrete_mapping:
            # then the queries except the last propensity score are not actual values but index.
            query_setting = []
            query_setting_grad = []
            for i in range(len(self.discrete_mapping)):
                # query_setting.append(self.discrete_mapping[i][int(query[i])])
                query_setting.append(
                    (self.discrete_mapping[i](int(query[i]))))
                if action_grad is not None:
                    query_setting_grad.append(
                        (torch.Tensor(self.discrete_mapping_grad[i]).cuda()*action_grad[i]).sum())

            if idx_grad is not None:
                idx_setting_grad = torch.matmul(idx_grad, torch.Tensor(
                    list(self.obj.values())).cuda())
                query_setting_grad = torch.cat([
                    idx_setting_grad, torch.stack(query_setting_grad)])

        else:
            query_setting = list(query)

        # Assume the cluster of expensive, obj-related variables are before the cluster of cheap, RL-chosen variables.
        setting_value = self.obj[idx]+query_setting
        inputs, targets = self.simulator.simulate(np.array([setting_value]))
        if self.args.noise:
            # TODO:
            pass
        trajectory = merge_inputs_targets_onehot(inputs, targets)
        return trajectory, query_setting, query_setting_grad

    def process_new_data(self, action, new_datapoint, intervene):
        """Do intervention on the rel_graph while check repeatance of the action

        Args:
            action ([type]): [description] contains gradient
            new_datapoint ([type]): [description] rollout of the action, no gradient.
        """
        repeat = 0
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
                new_setting = new_datapoint[0, :, 0, 0]
                for l in self.obj_data.values():
                    if l:
                        for d in l:
                            setting = d[0, :, 0, 0]
                            if intervene:
                                # found a single perfect intervention
                                if (new_setting-setting != 0).sum() == 1:
                                    a = torch.abs(new_setting-setting)
                                    b = torch.abs(
                                        new_datapoint[0, :, -1, 0]-d[0, :, -1, 0])
                                    # print('a', a, 'b', b)
                                    causal = torch.nonzero(a).item()
                                    caused = torch.nonzero(b).flatten()
                                    if a[causal] == b[causal]:
                                        caused = caused[caused != causal]
                                    # print('caused', caused, 'causal', causal)
                                    if len(caused.numpy()) > 0:
                                        self.causal_model.rel_graph[0,
                                                                    0, caused*self.args.num_atoms+causal, 0] = 0
                                        self.causal_model.rel_graph[0,
                                                                    0, caused*self.args.num_atoms+causal, 1] = 100

                            if (new_setting-setting == 0).sum() == self.args.num_atoms:
                                repeat += 1

                                # causal = abs(
                                #     new_setting-setting).unsqueeze(0)
                                # caused = abs(
                                #     new_datapoint[0, :, -1, 0]-d[0, :, -1, 0]).unsqueeze(1).cuda()
                                # relations = (causal*caused).flatten()
                                # import pdb
                                # pdb.set_trace()
                                # self.causal_model.rel_graph[0, 0, :,
                                #                             0] -= relations*5
                                # self.causal_model.rel_graph[0, 0, :,
                                # 1] += relations*5
        return repeat

    def step(self, idx, action, new_datapoint):  # :action, memory):
        """[summary]

            Args:
                action ([type]): the obj index and the new trajectory setting action encoded with RL agent's notation.

            Returns:
                state: [learning_assess, obj_data_features]
                reward: - val_MSE
                done: Whether the data budget is met. If yes, the training can end early.
            """
        # new_datapoint is a entire rollout trajectory.
        new_datapoint = torch.Tensor(new_datapoint)
        # action should have gradient, while new_datapoint doesn't
        repeat = self.process_new_data(
            action, new_datapoint, self.args.intervene)
        # print('3')
        # GPUtil.showUtilization()
        # self.train_dataset.data size (batch_size, num_nodes, timesteps, feat_dims)
        self.obj_data[idx].append(new_datapoint)
        self.train_dataset.update(new_datapoint)
        # print(self.train_dataset.data[:, :, 0, 0])
        train_data_loader = DataLoader(
            self.train_dataset, batch_size=self.args.train_bs, shuffle=False)
        for i in range(self.args.epochs):
            print(str(i), "iter of epoch", self.epoch)
            nll, nll_lasttwo, kl, mse, control_constraint_loss, lr, rel_graphs, rel_graphs_grad, a, b, c, d, e, f = train_control(
                self.args, self.log_prior, self.causal_model_optimizer, self.save_folder, train_data_loader, self.causal_model, self.epoch)

        if self.epoch % self.args.train_log_freq == 0:
            self.logger.log('train', self.causal_model, self.epoch, nll, nll_lasttwo, kl=kl, mse=mse, control_constraint_loss=control_constraint_loss, lr=lr, rel_graphs=rel_graphs,
                            rel_graphs_grad=rel_graphs_grad, msg_hook_weights=a, nll_train_lasttwo=b, nll_train_lasttwo_5=c, nll_train_lasttwo_10=d, nll_train_lasttwo__1=e, nll_train_lasttwo_1=f)

        # val_loss = val_control(
        #     self.args, self.log_prior, self.logger, self.save_folder, self.valid_data_loader, self.epoch, self.causal_model, self.scheduler)
        self.scheduler.step()
        val_loss = 0

        state = self.extract_features()
        # TODO: the penalty may also need to punish repeated queries.
        penalty = float(val_loss)+10*self.train_dataset.data.size(0)+100*repeat
        done = self.train_dataset.data.size(0) > self.args.budget
        self.epoch += 1
        print('repeat', repeat, 'val_loss', val_loss,
              'self.train_dataset.data.size(0)', self.train_dataset.data.size(0))
        return state, -penalty, done

    def extract_features(self):
        # batch_size, num_nodes, step, dim
        # TODO: change it to include one hot feature
        pred = self.causal_model(self.learning_assess_data)[0][:, :, :, 0]
        pred = pred.transpose(1, 2).detach()
        # learning_assess = self.learning_extractor(pred)
        learning_assess = pred
        learning_assess_feat = torch.zeros(
            (learning_assess.size(0), 2*self.feature_dims)).cuda()
        for k in range(learning_assess.size(0)):
            out, [h, c] = self.learning_assess_extractor(
                learning_assess[k:k+1])
            unpack = out.view(-1, 1, self.lstm_direction, self.feature_dims)
            # (batch_size, hidden_size)
            learning_assess_feat[k:k+1] = torch.cat(
                (unpack[-1, :, 0, :], unpack[0, :, 1, :]), axis=1)

        # (1, 2 * feature_dims)
        set_learning_assess_feat = self.deepset(
            learning_assess_feat, axis=0, keepdim=False)
        # set_learning_assess_feat = learning_assess_feat.sum(0, keepdim=True)
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass

        obj_data_features = []
        for i in self.obj.keys():
            obj_param = torch.Tensor(self.obj[i]).cuda()
            # TODO: change it to include one hot feature
            # (batch, step, num_nodes), batch_first inputs
            obj_traj = torch.cat(self.obj_data[i]).cuda()[
                :, :, :, 0].transpose(1, 2)

            all_traj_feat = torch.zeros(
                (obj_traj.size(0), 2*self.feature_dims)).cuda()
            for j in range(obj_traj.size(0)):
                # (seq_len, batch, num_directions * hidden_size)
                out, [h, c] = self.obj_data_extractor(obj_traj[j:j+1, :, :])
                unpack = out.view(-1, 1, self.lstm_direction,
                                  self.feature_dims)
                # (batch_size, 2 * hidden_size)
                all_traj_feat[j:j +
                              1] = torch.cat((unpack[-1, :, 0, :], unpack[0, :, 1, :]), axis=1)

            set_traj_feat = self.deepset(all_traj_feat, axis=0, keepdim=True)
            # (1, 3 * feature_dims)
            obj_data_features.append(
                torch.cat([self.obj_extractor(obj_param).unsqueeze(0), set_traj_feat], axis=1))

        # obj_data_features = [torch.zeros((1, 3*self.feature_dims)).cuda()]*6
        del out, h, c, unpack, learning_assess, learning_assess_feat, obj_param, obj_traj, all_traj_feat, set_traj_feat, pred
        return [set_learning_assess_feat, obj_data_features]
        # obj_data_features=torch.cat(obj_data_features).flatten()
        # return torch.cat([set_learning_assess_feat,obj_data_features])

    def extract_features_1(self):
        set_learning_assess_feat = torch.zeros((1, 128)).cuda()
        obj_data_features = [torch.zeros((1, 192)).cuda()]*6
        return [set_learning_assess_feat, obj_data_features]

    def reset(self, data_num_per_obj=1):
        # TODO: for fixed object case, no need to reinit self.obj
        # self.obj = {}
        self.obj_data = {i: [] for i in range(self.obj_num)}
        self.init_train_data(data_num_per_obj=data_num_per_obj)
        train_data = []
        for i in self.obj_data.values():
            for j in i:
                train_data.append(j)
        self.train_dataset = RLDataset(
            torch.cat(train_data), self.edge, self.mins, self.maxs)
        return self.extract_features()
