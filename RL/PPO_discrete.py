import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils.functions import gumbel_softmax
from data.datasets import RLDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
valid_data=[]
np.random.seed(10)
for i in range(320):
    t=[0]
    t.extend(np.random.randint(6,size=3).tolist())
    valid_data.append(t)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self): 
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, args, action_values):
        super(ActorCritic, self).__init__()
        self.args = args
        # actor
        self.action_feature = nn.Sequential(
            nn.Linear(5*args.extract_feat_dim, self.args.rl_hidden),
            nn.Tanh(),
            nn.Linear(self.args.rl_hidden, self.args.rl_hidden),
            nn.Tanh()
        ).cuda()

        # Since each action type is deciding by its own player, assuming indepence. TODO: fix an order and make it bayesian.
        # For propensity score
        self.action_player = [nn.Linear(self.args.rl_hidden, 1).cuda()]
        for i in range(self.args.action_dim):
            # maybe more layers
            self.action_player.append(
                nn.Linear(self.args.rl_hidden, len(action_values[i+1])).cuda())
        self.softmax = nn.Softmax(dim=0).cuda()

        # critic
        self.value_player = nn.Sequential(
            nn.Linear(3*self.args.extract_feat_dim *
                      self.args.initial_obj_num+2*self.args.extract_feat_dim, self.args.rl_hidden),
            nn.Tanh(),
            nn.Linear(self.args.rl_hidden, self.args.rl_hidden),
            nn.Tanh(),
            nn.Linear(self.args.rl_hidden, 1)
        ).cuda()

    def forward(self):
        raise NotImplementedError

    def act(self, memory, state):
        """
        Object-wise setting, then select one setting by argmax
        state: [set_learning_assess_feat, obj_data_features]. obj_data_features contains first obj feature then obj data feature.
        """
        state = state.squeeze()
#         set_learning_assess_feat = state[:2*self.args.extract_feat_dim]
#         obj_data_features = state[2 *
#                                   self.args.extract_feat_dim:].view(-1, 3*self.args.extract_feat_dim)
#         num_obj = obj_data_features.size(0)

#         all_state = torch.cat([
#             set_learning_assess_feat.expand(num_obj, -1), obj_data_features], dim=-1)

        all_state_feat = self.action_feature(all_state)
        # find the most_wanted setting by first dimension.
#         propensity = self.softmax(self.action_player[0](all_state_feat))[:, 0]
#         dist = Categorical(propensity)
#         most_wanted = dist.sample()
#         most_wanted_idx = most_wanted.item()
#         # The others do not matter anymore so not updating their log probs.
#         action_logprob = dist.log_prob(most_wanted)
#         dist_entropy = dist.entropy()
#         complete_action = [most_wanted_idx]
        
        most_wanted_idx=0
        dist_entropy =0
        action_logprob =0
        complete_action = [most_wanted_idx]
    

        for an in range(1, self.args.action_dim+1):
            action_probs = self.softmax(
                self.action_player[an](all_state_feat[most_wanted_idx]))
            dist = Categorical(action_probs)
            action = dist.sample()
            complete_action.append(action.item())
            action_logprob += dist.log_prob(action)
            dist_entropy += dist.entropy()

        memory.states.append(state)
        memory.actions.append(torch.Tensor(complete_action))
        memory.logprobs.append(action_logprob)
        return complete_action

    def evaluate(self, states, actions):
        """
        Evaluate the choice for the ith action type
        """
        num_data = states.size(0)
        action_logprobs, state_values, dist_entropys = torch.zeros((num_data)).cuda(
        ), torch.zeros((num_data)).cuda(), torch.zeros((num_data)).cuda(),
        for i in range(num_data):
            state = states[i]
            action = actions[i]
            set_learning_assess_feat = state[:2*self.args.extract_feat_dim]
            obj_data_features = state[2 *
                                      self.args.extract_feat_dim:].view(-1, 3*self.args.extract_feat_dim)
            num_obj = obj_data_features.size(0)

            all_state = torch.cat([
                set_learning_assess_feat.expand(num_obj, -1), obj_data_features], dim=-1)

            all_state_feat = self.action_feature(all_state)
            # find the most_wanted setting by last dimension.
#             propensity = self.softmax(
#                 self.action_player[0](all_state_feat))[:, 0]
#             dist = Categorical(propensity)
#             action_logprob = dist.log_prob(action[0])
#             dist_entropy = dist.entropy()

            dist_entropy =0
            action_logprob =0

            for an in range(1, self.args.action_dim+1):
                action_probs = self.softmax(
                    self.action_player[an](all_state_feat[int(action[0])]))
                dist = Categorical(action_probs)
                action_logprob += dist.log_prob(action[an])
                dist_entropy += dist.entropy()

            # action_logprobs and dist_entropy size: (batch_size)
            # state_value size: (batch_size, 1)
            state_value = self.value_player(state)

            action_logprobs[i] = action_logprob
            state_values[i] = state_value
            dist_entropys[i] = dist_entropy

        return action_logprobs, state_values, dist_entropys


class PPO:
    def __init__(self, args, discrete_mapping, betas, eps_clip):
        self.gamma = args.rl_gamma
        self.eps_clip = eps_clip
        self.K_epochs = args.K_epochs

        self.policy = ActorCritic(args, discrete_mapping).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=args.rl_lr, betas=betas)
        self.policy_old = ActorCritic(args, discrete_mapping).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.count = 0
        self.update_count = 0
        
    def validate_state(self, env):
        """
        Can only be called when there's no loss in env.reset()
        valid_data: [action indices, rewards, is_terminals]
        """
        def mini_reset():
            env.obj = {i: env.discrete_mapping[0][i]
                        for i in range(env.args.initial_obj_num)}
            env.obj_data = {i: [] for i in range(env.obj_num)}
            env.train_dataset = RLDataset(
            torch.Tensor(), env.edge, env.mins, env.maxs)

            new_datapoint, query_setting, _ = env.action_to_new_data([0,3,0,5])
            env.obj_data[0].append(new_datapoint)
            env.train_dataset.update(new_datapoint.clone())
        
        memory = Memory()
        mses = []
        mini_reset()
        state = env.extract_features()
        for j in range(len(valid_data)):
            complete_action = valid_data[j]
            new_datapoint, query_setting, _ = env.action_to_new_data(
                        complete_action)
            repeat, num_intervention = env.process_new_data(
                        complete_action, new_datapoint, env.args.intervene)       
            idx = int(complete_action[0])
            env.obj_data[idx].append(new_datapoint)
            env.train_dataset.update(new_datapoint.clone())
            
            memory.states.append(state.squeeze())
            memory.actions.append(torch.Tensor(complete_action))
#             print(len(env.train_dataset))
            state, reward, done = env.step_entropy(num_intervention)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            if done:
                mini_reset()
            
         
        discounted_reward = 0
        rewards = []
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
                
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
#         import pdb;pdb.set_trace()
            
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        dataset = TensorDataset(old_states, old_actions, rewards)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        for batch_idx, (old_states, old_actions, rewards) in enumerate(loader):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            mse = self.MseLoss(state_values, rewards)
            mses.append(mse.item())
            
        env.logger.log_arbitrary(self.count,
                                 SRMSE_val=np.mean(mses))
        print('[PPO] SRMSE_val logged!')
            
        



    def update(self, env, memory):
        # Update the action and value player controling the ith action type
        # Monte Carlo estimate of state rewards:
        self.update_count += 1
        rewards = []
        discounted_reward = np.inf
        episode_rewards=[]
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                if discounted_reward != np.inf:
                    episode_rewards.append(discounted_reward)
                discounted_reward = 0
                
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        episode_rewards.append(discounted_reward)
        
        env.logger.log_arbitrary(self.update_count,
                                 RL_75_episode_rewards=np.percentile(episode_rewards, 75), 
                                 RL_50_episode_rewards=np.percentile(episode_rewards, 50),
                                 RL_25_episode_rewards=np.percentile(episode_rewards, 25))
#         import pdb;pdb.set_trace()

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        dataset = TensorDataset(old_states, old_actions, old_logprobs, rewards)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        # Optimize policy for K epochs:
        for i in range(self.K_epochs):
#             print(i)
            for batch_idx, (old_states, old_actions, old_logprobs, rewards) in enumerate(loader):
                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    old_states, old_actions)

                # Finding the ratio (pi_theta / pi_theta_old):
                ratios = torch.exp(logprobs - old_logprobs.detach())
#                 print('ratios', ratios[:20])
#                 print('rewards', rewards[:20])
#                 print('state_values', state_values[:20])

                # Finding Surrogate Loss:
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                    1+self.eps_clip) * advantages
                mse = self.MseLoss(state_values, rewards)
                print('state_values', state_values)
                print('rewards', rewards)
                env.f.write(str(state_values))
                env.f.write('\n')
                env.f.write(str(rewards))
                env.f.write('\n')
                env.f.flush()
                loss = -torch.min(surr1, surr2) + mse - 0.01*dist_entropy
                # print(surr1, surr2, state_values, rewards, dist_entropy)

                # take gradient step
                self.optimizer.zero_grad()
                if env.feature_extractors:
                    env.obj_extractor_optimizer.zero_grad()
                    env.obj_data_extractor_optimizer.zero_grad()
                    env.learning_assess_extractor_optimizer.zero_grad()

                loss.mean().backward()

                self.optimizer.step()
                # TODO: Doing update for both policy and state representation simultaneously. May change to 2 step in the future using args.extractors-update-epoch.
                if env.feature_extractors:
                    env.obj_extractor_optimizer.step()
                    env.obj_data_extractor_optimizer.step()
                    env.learning_assess_extractor_optimizer.step()
                env.logger.log_arbitrary(self.count,
                                 RL_loss=loss.mean().item(),
                                 RL_surr=torch.min(surr1, surr2).mean().item(),
                                 RL_entropy=dist_entropy.mean().item(), 
                                 RL_SRMSE=mse.mean().item())
                                         
                self.count += 1
        print('[PPO] train done.')
        # After updating all action types, copy new weights into old policy:
#         self.policy_old.load_state_dict(self.policy.state_dict())
