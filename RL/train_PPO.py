import torch
from torch.distributions import Categorical
# from utils.functions import gumbel_softmax


def train_rl(env, memory, ppo):
    # render = False
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    for i_episode in range(1, env.args.rl_epochs+1):
        env.reset()
        state = env.extract_features()
        episode_penalty = 0
        for t in range(env.args.rl_max_timesteps):
            timestep += 1
            complete_action_grad, complete_action = ppo.policy_old.act(
                memory, state)

            # TODO: get requires_grad working with 1 level of action.
            if env.args.action_requires_grad:
                temp = []
                for i in obj_queries_grad:
                    temp.append(torch.stack(i[:-1]).unsqueeze(0).cuda())
                most_wanted_query_grad = torch.cat(temp, axis=0).permute(
                    1, 2, 0)*idx_onehot.cuda()
                most_wanted_query_grad = most_wanted_query_grad.permute(
                    2, 0, 1).sum(0)
                # actual numbers
                idx, new_datapoint, query_setting, query_setting_grad = env.action_to_new_data(
                    [idx, most_wanted_query], idx_grad=idx_onehot, action_grad=most_wanted_query_grad)
                state, reward, done = env.step(
                    idx, query_setting_grad, new_datapoint)
            else:
                idx, new_datapoint, query_setting, _ = env.action_to_new_data(
                    complete_action)
                repeat = env.process_new_data(
                    complete_action, new_datapoint, env.args.intervene)
                val_loss = env.train_causal(
                    idx, query_setting, new_datapoint)
                state, reward, done = env.step(val_loss)
                print('repeat', repeat, 'intervened nodes', env.intervened_nodes, 'val_loss', val_loss,
                      'self.train_dataset.data.size(0)', env.train_dataset.data.size(0))
                env.logger.log_arbitrary(env.epoch,
                                         RLAL_train_dataset_size=env.train_dataset.data.size(
                                             0),
                                         RLAL_repeat=repeat,
                                         RLAL_num_intervention=len(env.intervened_nodes), RLAL_penalty=-reward,
                                         RLAL_causal_converge=env.early_stop_monitor.stopped_epoch)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % env.args.rl_update_timestep == 0:
                ppo.update(env, memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            episode_penalty += -reward
            if done:
                break

        avg_length += t

        # logging
        env.logger.log_arbitrary(i_episode, episode_penalty=episode_penalty)
        torch.save(ppo.policy.state_dict(), '_'.join(
            [env.args.save_folder, 'PPO_{}.pth'.format(i_episode)]))
        if i_episode % env.args.rl_log_freq == 0:
            avg_length = int(avg_length/env.args.rl_log_freq)
            running_reward = int((running_reward/env.args.rl_log_freq))

            print('Episode {} \t avg length: {} \t reward: {}'.format(
                i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
