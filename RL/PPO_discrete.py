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
    def __init__(self, state_dim, action_dim, n_latent_var, action_num):
        # action_num: number of action types, each has dimension=action_dim. +1 because of propensity score.
        super(ActorCritic, self).__init__()
        # actor
        self.action_feature = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh()
        ).cuda()

        self.action_num = action_num
        # Since each action type is deciding by its own player, assuming indepence. TODO: fix an order and make it bayesian.
        self.action_player = []
        for i in range(self.action_num-1):
            self.action_player.append(
                nn.Linear(n_latent_var, action_dim).cuda())
        self.action_player.append(nn.Linear(n_latent_var, 1).cuda())
        self.softmax = nn.Softmax(dim=-1).cuda()

        # critic
        self.value_player = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        ).cuda()

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """
        Return the choice for the ith action type
        """
        # state = torch.from_numpy(state).float().to(device)
        complete_action = []
        complete_action_grad = []

        action_logprobs = 0
        state_feat = self.action_feature(state)
        for i in range(self.action_num-1):
            # action_probs = self.softmax(self.action_player[i](state_feat))
            action_onehot, action_probs = gumbel_softmax(
                self.action_player[i](state_feat), hard=True)
            # dist = Categorical(action_probs)
            # action = dist.sample()
            # action_logprobs += dist.log_prob(action)
            action_logprobs += torch.log((action_probs*action_onehot).sum())
            action = action_onehot[0]
            complete_action.append(action.argmax().item())
            complete_action_grad.append(action)
        # The last dimension is propensity score.
        complete_action.append(self.action_player[-1](state_feat).item())
        complete_action_grad.append(self.action_player[-1](state_feat)[0])
        return complete_action_grad, complete_action, action_logprobs

    def evaluate(self, state, action):
        """
        Evaluate the choice for the ith action type
        """
        action_logprobs = 0
        dist_entropy = 0

        state_feat = self.action_feature(state)
        for i in range(self.action_num-1):
            action_probs = self.action_layer[i](state_feat)
            dist = Categorical(action_probs)
            action_logprobs += dist.log_prob(action)
            dist_entropy += dist.entropy()
        # Also need to add the propensity score here

        state_value = self.value_player(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, action_num):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(
            state_dim, action_dim, n_latent_var, action_num).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(
            state_dim, action_dim, n_latent_var, action_num).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
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
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta_old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # After updating all action types, copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
