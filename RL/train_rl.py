import torch
from torch.distributions import Categorical
from utils.functions import gumbel_softmax


def train_rl(env, memory, ppo):
    # render = False
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    for i_episode in range(1, env.args.rl_epochs+1):
        state = env.reset()
        for t in range(env.args.rl_max_timesteps):
            timestep += 1
            memory.states.append(state)

            learning_assess, obj_data_features = state
            obj_queries = []
            obj_logprobs = []
            obj_queries_grad = []

            for i in range(env.obj_num):
                complete_feat = torch.cat(
                    [learning_assess, obj_data_features[i]], axis=1)
                complete_action_grad, complete_action, action_logprobs = ppo.policy_old.act(
                    complete_feat)
                obj_queries.append(complete_action)
                obj_logprobs.append(action_logprobs)
                obj_queries_grad.append(complete_action_grad)

            # Use gumbel softmax to replace categorical sampling.
            # obj_prob = (torch.Tensor(obj_queries)[:, -1]).softmax(0)
            # Not deterministic
            # idx = Categorical(obj_prob).sample().item()

            obj_select_grad = []
            for i in range(len(obj_queries_grad)):
                obj_select_grad.append(obj_queries_grad[i][-1])

            idx_onehot, idx_probs = gumbel_softmax(
                torch.cat(obj_select_grad), hard=True)
            # print('out func', idx_onehot.requires_grad, idx_probs.requires_grad)
            idx = idx_onehot.argmax().item()
            most_wanted_query = obj_queries[idx]
            if env.args.action_requires_grad:
                temp = []
                for i in obj_queries_grad:
                    temp.append(torch.stack(i[:-1]).unsqueeze(0).cuda())
                most_wanted_query_grad = torch.cat(temp, axis=0).permute(
                    1, 2, 0)*idx_onehot.cuda()
                most_wanted_query_grad = most_wanted_query_grad.permute(
                    2, 0, 1).sum(0)
                # actual numbers
                new_datapoint, query_setting, query_setting_grad = env.action_to_new_data(
                    [idx, most_wanted_query], idx_grad=idx_onehot, action_grad=most_wanted_query_grad)
                state, reward, done = env.step(
                    idx, query_setting_grad, new_datapoint)
            else:
                new_datapoint, query_setting, _ = env.action_to_new_data(
                    [idx, most_wanted_query])
                state, reward, done = env.step(
                    idx, query_setting, new_datapoint)

            memory.actions.append([idx]+query_setting)
            # add the idx logprob
            memory.logprobs.append(obj_logprobs[idx]+torch.log(idx_probs))
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % env.args.rl_update_timestep == 0:
                ppo.update(env, memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            # if render:
            #     env.render()
            if done:
                break

        avg_length += t

        # stop the entire training (end episodes) if avg_reward > solved_reward
        if running_reward > (env.args.rl_log_freq*env.args.solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(),
                       './PPO_{}.pth'.format(env_name))
            break

        # logging
        if i_episode % env.args.rl_log_freq == 0:
            avg_length = int(avg_length/env.args.rl_log_freq)
            running_reward = int((running_reward/env.args.rl_log_freq))

            print('Episode {} \t avg length: {} \t reward: {}'.format(
                i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
