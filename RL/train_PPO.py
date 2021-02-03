import os
import numpy as np
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
        env.f.write(''.join(['\nepisode', str(i_episode), '\n']))
        env.f.flush()

        for t in range(env.args.rl_max_timesteps):
            timestep += 1
            complete_action = ppo.policy_old.act(
                memory, state)
#             first=env.ind_dict[str(env.fff[env.fff_count][:3])]
#             complete_action=[first]+env.fff[env.fff_count][3:]
#             env.fff_count +=1

            env.f.write(str(env.ind_dict_rev[complete_action[0]])+' '+str(complete_action[1:]))
            env.f.write('\n')
            env.f.flush()

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
                new_datapoint, query_setting, _ = env.action_to_new_data(
                    complete_action)
                repeat, num_intervention = env.process_new_data(
                    complete_action, new_datapoint, env.args.intervene)
                
                idx = int(complete_action[0])
                env.obj_data[idx].append(new_datapoint)
                env.train_dataset.update(new_datapoint.clone())
#                 print('data for object', idx, np.stack(env.obj_data[idx])[:, 0, :, 0, 0])
                
#                 val_loss = env.train_causal()
                val_loss = 0
                env.epoch += 1
                state, reward, done = env.step_entropy(num_intervention)
#                 state, reward, done = env.step(val_loss)
                print('repeat', repeat, 'intervened nodes', env.intervened_nodes,
                      'val_loss', val_loss, 'total interventions', env.total_intervention,
                      'penalty', -reward, 'self.train_dataset.data.size(0)', env.train_dataset.data.size(0))
                print('action', complete_action)
                env.logger.log_arbitrary(env.epoch,
                                         RLAL_repeat=repeat,
                                         RLAL_interventioned_nodes=len(env.intervened_nodes), 
                                         RLAL_total_intervention=env.total_intervention,
                                         RLAL_penalty=-reward,
                                         RLAL_causal_converge=env.early_stop_monitor.stopped_epoch)
        
                env.f.write(str(env.intervened_nodes))
                env.f.write('\n')
                env.f.flush()

            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            # update if its time
            if timestep % env.args.rl_update_timestep == 0:
                ppo.update(env, memory)
                memory.clear_memory()
                torch.save(ppo.policy.state_dict(), os.path.join(
                    env.logger.save_folder, 'PPO_{}.pth'.format(str(timestep))))
                others={}
                others['action_players']=ppo.player_params
                others['obj_extractor']=list(env.obj_extractor.parameters())
                others['obj_data_extractor']=list(env.obj_data_extractor.model.parameters())+\
                [env.obj_data_extractor.h0, env.obj_data_extractor.c0]
                others['learning_assess_extractor']=list(env.learning_assess_extractor.parameters())+\
                [env.learning_assess_extractor.h0, env.learning_assess_extractor.c0]
                torch.save(others, os.path.join(
                    env.logger.save_folder, 'PPO_others_{}.pth'.format(str(timestep))))
#                 timestep = 0

#             if timestep % env.args.rl_update_timestep == 0:
                print('[PPO] weights updated.')
#                 ppo.policy_old.load_state_dict(ppo.policy.state_dict())
#                 env.obj_extractor.load_state_dict(env.obj_extractor_new.state_dict())
#                 env.obj_data_extractor.load_state_dict(env.obj_data_extractor_new.state_dict())
#                 env.learning_assess_extractor.load_state_dict(env.learning_assess_extractor_new.state_dict())
#                 ppo.validate_state(env)

            running_reward += reward
            if done:
                print('[PPO] done.')
                break

        env.f.write('total_intervention '+str(env.total_intervention)+'\n')
        env.f.flush()
        
        avg_length += t
        env.logger.log_arbitrary(i_episode,
                                 RLAL_episode_train_dataset_size=env.train_dataset.data.size(
                                     0),
                                 RLAL_episode_interventioned_nodes=len(env.intervened_nodes), 
                                 RLAL_episode_total_intervention=env.total_intervention,
#                                  RLAL_episode_penalty=-reward,
                                 RLAL_episode_length=t)
        
        if i_episode % env.args.rl_log_freq == 0:
                
            avg_length = int(avg_length/env.args.rl_log_freq)
            running_reward = int((running_reward/env.args.rl_log_freq))

            print('Episode', i_episode,  'avg length', avg_length, 'reward', running_reward)
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
