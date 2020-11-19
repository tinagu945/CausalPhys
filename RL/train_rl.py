import torch
from torch.distributions import Categorical
from utils.functions import gumbel_softmax


def train_rl(args, env, memory, ppo):
    ############## Hyperparameters ##############
    state_dim = 128
    # Number of value options for one attribute
    action_dim = 6
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    update_timestep = 2000      # update policy every n timesteps
    #############################################

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    for i_episode in range(1, args.epochs+1):
        state = env.reset()
        for t in range(max_timesteps):
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

            # Running policy_old:
            # Last col gives "confidence" of the query
            # obj_prob = (torch.Tensor(obj_queries)[:, -1]).softmax(0)
            # Not deterministic
            # idx = Categorical(obj_prob).sample().item()
            obj_select_grad = []
            for i in range(len(obj_queries_grad)):
                obj_select_grad.append(obj_queries_grad[i][-1])

            idx_onehot, idx_probs = gumbel_softmax(
                torch.cat(obj_select_grad), hard=True)
            print('out func', idx_onehot.requires_grad, idx_probs.requires_grad)
            idx = idx_onehot.argmax().item()
            # indices
            most_wanted_query = obj_queries[idx]
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
            memory.actions.append([idx]+query_setting)
            # add the idx logprob
            memory.logprobs.append(obj_logprobs[idx]+torch.log(idx_probs))
            state, reward, done = env.step(
                args, idx, query_setting_grad, new_datapoint)
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            # if render:
            #     env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(),
                       './PPO_{}.pth'.format(env_name))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(
                i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
