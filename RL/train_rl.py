import torch
from torch.distributions import Categorical


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
            for i in range(len(obj_data_features)):
                complete_feat = torch.cat(
                    [learning_assess, obj_data_features[i]], axis=1)
                complete_action, action_logprobs = ppo.policy_old.act(
                    complete_feat)
                obj_queries.append(complete_action)
                obj_logprobs.append(action_logprobs)

            # Running policy_old:
            # Last col gives "confidence" of the query
            obj_prob = (torch.Tensor(obj_queries)[:, -1]).softmax(0)
            # Not deterministic
            idx = Categorical(obj_prob).sample().item()
            most_wanted_query = obj_queries[idx]

            new_datapoint, query_setting = env.action_to_new_data(
                [idx, most_wanted_query])
            memory.actions.append([idx]+query_setting)
            memory.logprobs.append(obj_logprobs[idx])
            state, reward, done = env.step(args, idx, new_datapoint)
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
