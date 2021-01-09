import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym

from utils.functions import gumbel_softmax

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        set_learning_assess_feat = state[:2*self.args.extract_feat_dim]
        obj_data_features = state[2 *
                                  self.args.extract_feat_dim:].view(-1, 3*self.args.extract_feat_dim)
        num_obj = obj_data_features.size(0)

        all_state = torch.cat([
            set_learning_assess_feat.expand(num_obj, -1), obj_data_features], dim=-1)

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
        
        most_wanted_idx=200
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
        self.gamma = args.gamma
        self.eps_clip = eps_clip
        self.K_epochs = args.K_epochs

        self.policy = ActorCritic(args, discrete_mapping).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=args.rl_lr, betas=betas)
        self.policy_old = ActorCritic(args, discrete_mapping).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, env, memory):
        # Update the action and value player controling the ith action type
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # was 1e-5
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-3)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for i in range(self.K_epochs):
#             print(i)
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)
            import pdb;pdb.set_trace()

            # Finding the ratio (pi_theta / pi_theta_old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            # print(surr1, surr2, state_values, rewards, dist_entropy)

            # take gradient step
            self.optimizer.zero_grad()
#             env.obj_extractor_optimizer.zero_grad()
#             env.obj_data_extractor_optimizer.zero_grad()
#             env.learning_assess_extractor_optimizer.zero_grad()

            loss.mean().backward()

            self.optimizer.step()
            # TODO: Doing update for both policy and state representation simultaneously. May change to 2 step in the future using args.extractors-update-epoch.
#             env.obj_extractor_optimizer.step()
#             env.obj_data_extractor_optimizer.step()
#             env.learning_assess_extractor_optimizer.step()

        # After updating all action types, copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
