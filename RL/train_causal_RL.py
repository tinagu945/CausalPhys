"""Train data like Interpretable Physics"""
import pdb
import time
import argparse
import os
import datetime
import sys
import itertools
import copy

from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.functions import *
from RL.train_PPO import train_rl
# from  import val_control
# from test import test_control
from utils.logger import Logger
from AL.AL_control_sampler import RandomPytorchSampler
from data.datasets import *
from data.dataset_utils import *
from data.generate_dataset import generate_dataset_discrete, get_noise_std
from RL.PPO_discrete import *
from AL.AL_env import *
from AL.AL_env_entropy import *
from data.simulator import RolloutSimulator
from data.scenarios import FrictionSliding
from utils.general_parser import general_parser

parser = general_parser()
parser.add_argument('--rl-epochs', type=int, default=1000,
                    help='#epochs for each PPO memory training.')
parser.add_argument('--patience', type=int, default=5,
                    help='Number of epochs after which if validation error has not decreased, we stop the training.')
parser.add_argument('--rl-log-freq', type=int, default=1,
                    help='How many epochs every logging for rl training.')
# parser.add_argument('--solved-reward', type=float, default=-500,
#                     help='Stop the entire training (end episodes) of PPO if avg_reward > solved_reward')
parser.add_argument('--extractors-update-epoch', type=int, default=20,
                    help='How many epochs every gradient descent for feature extractors.')
parser.add_argument('--rl-update-timestep', type=int, default=20,
                    help='How many epochs every gradient descent for PPO policy.')
parser.add_argument('--rl-max-timesteps', type=int, default=1010,
                    help='How many times the PPO policy can try for each episode.')
parser.add_argument('--rl-lr', type=float, default=0.002,
                    help='lr to train the rl policy')
parser.add_argument('--obj-extractor-lr', type=float, default=1e-3,
                    help='lr to train the obj_extractor')
parser.add_argument('--obj-data-extractor-lr', type=float, default=1e-3,
                    help='lr to train the obj_data_extractor')
parser.add_argument('--learning-assess-extractor-lr', type=float, default=1e-3,
                    help='lr to train the learning_assess_extractor')

parser.add_argument('--rl-gamma', type=float, default=0.99,
                    help='discount factor of the rl policy')
parser.add_argument('--rl-hidden', type=int, default=64,
                    help='Number of hidden units for the policy network.')
parser.add_argument('--extract-feat-dim', type=int, default=32,
                    help='Hidden dimension for obj_extractor, obj_data_extractor, learning_extractor and learning_assess_extractor.')
parser.add_argument('--budget', type=int, default=1000,
                    help='If the causal model queried for more data than budget, env reset.')
parser.add_argument('--initial-obj-num', type=int, default=216,
                    help='Number of objects available at the beginning.')
parser.add_argument('--noise', type=float, default=None,
                    help='The noise the data simulator adds to training data.')
parser.add_argument('--causal-threshold', type=float, default=0,
                    help='The threshold above which it is possible a causal relation does exist under an intervention.')
parser.add_argument('--action-requires-grad', action='store_true', default=False,
                    help='Whether the action needs gradient for intervene_graph.')
parser.add_argument('--intervene', action='store_true', default=False,
                    help='Whether do the intervention when there exists paired data.')

args = parser.parse_args()
assert args.train_bs == args.val_bs
args.num_atoms = args.input_atoms+args.target_atoms
args.script = 'RL_PPO'
if args.gt_A or args.all_connect:
    print('Using given graph and kl loss will be omitted')
    args.kl = 0
if args.noise:
    if args.causal_threshold == 0:
        print('You have a noise for rollout now but the noise causal_threshold you specified is 0!')
else:
    assert args.causal_threshold == 0

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Save model and meta-data. Always saves in a new sub-folder.
now = datetime.datetime.now()
timestamp = now.isoformat()
save_folder = '{}/exp{}/'.format(args.save_folder,
                                 '_'.join([args.script, timestamp]+[i.replace("--", "") for i in sys.argv[1:]]))
os.mkdir(save_folder)
meta_file = open(os.path.join(save_folder, 'meta.txt'), 'w')
print(args, file=meta_file)
print(save_folder)
meta_file.flush()

if args.self_loop:
    off_diag = np.ones([args.num_atoms, args.num_atoms])
else:
    off_diag = np.ones([args.num_atoms, args.num_atoms]) - \
        np.eye(args.num_atoms)

# This is not adjacency matrix since it's 49*7, not 7*7!
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec).cuda()
rel_send = torch.FloatTensor(rel_send).cuda()

prior = np.array([0.9, 0.1])  # TODO: hard coded for now
print("Using prior")
print(prior)
log_prior = torch.FloatTensor(np.log(prior))
log_prior = torch.unsqueeze(log_prior, 0)
log_prior = torch.unsqueeze(log_prior, 0).cuda()

logger = Logger(save_folder)
train_data_min_max = [[0, 0, 0, 0.53, 0, 0, 0, 0],
                      [1, 1, 0.56, 1.5, 1, 1, 381.24, 727.27]]
if args.val_grouped:
    # To see control loss, val and test should be grouped
    valid_data = OneGraphDataset.load_one_graph_data(
        'valid_causal_vel_'+args.val_suffix, train_data_min_max=train_data_min_max, size=args.val_size, self_loop=args.self_loop, control=True, control_nodes=args.input_atoms, variations=args.val_variations, need_grouping=args.val_need_grouping)
    valid_sampler = RandomPytorchSampler(valid_data)
    valid_data_loader = DataLoader(
        valid_data, batch_size=args.val_bs, shuffle=False, sampler=valid_sampler)

    # test_data = load_one_graph_data(
    #     'test_'+args.val_suffix, size=args.test_size, self_loop=args.self_loop, control=True, control_nodes=args.input_atoms, variations=4)
    # test_sampler = RandomPytorchSampler(test_data)
    # test_data_loader = DataLoader(
    #     test_data, batch_size=args.val_bs, shuffle=False, sampler=test_sampler)
else:
    valid_data = OneGraphDataset.load_one_graph_data(
        'valid_causal_vel_'+args.val_suffix, train_data_min_max=train_data_min_max, size=args.val_size, self_loop=args.self_loop, control=False)
    valid_data_loader = DataLoader(
        valid_data, batch_size=args.val_bs, shuffle=True)
    # test_data = load_one_graph_data(
    #     'test_'+args.val_suffix, size=args.val_size, self_loop=args.self_loop, control=False)
    # test_data_loader = DataLoader(
    #     test_data, batch_size=args.val_bs, shuffle=True)

betas = (0.9, 0.999)
eps_clip = 0.2

values = [[0.0, 0.2222222222222222, 0.3333333333333333, 0.7777777777777777, 0.8888888888888888, 1.0], [0.0, 0.3333333333333333, 0.4444444444444444, 0.5555555555555556,
                                                                                                       0.8888888888888888, 1.0], [0.0, 0.06222222222222223, 0.2488888888888889, 0.37333333333333335, 0.4977777777777778, 0.56]]

all_obj = [list(i) for i in list(itertools.product(*values))]
# all_obj = [[0, 1, 1], [1, 1, 1]]
assert args.initial_obj_num == len(all_obj)

discrete_mapping = [all_obj, [0.53, 0.6377777777777778, 0.7455555555555555, 1.1766666666666667, 1.2844444444444445, 1.5], [
    0.0, 0.1111111111111111, 0.4444444444444444, 0.7777777777777777, 0.8888888888888888, 1.0], [0.0, 0.1111111111111111, 0.3333333333333333, 0.4444444444444444, 0.5555555555555556, 1.0]]
# discrete_mapping = [all_obj, [0.53, 0.637], [0, 0.11], [0, 0.11]]
discrete_mapping_grad = [lambda x: 0.53+x *
                         (1.5-0.53)/5, lambda x: 0+x*(1-0)/5, lambda x: 0+x*(1-0)/5]
assert args.action_dim == len(discrete_mapping)-1
# action_dim: exclude objects
# discrete mapping: include objects

if 'init_data.npy' not in os.listdir(args.save_folder):
    decomposed_discrete_mapping=copy.deepcopy(discrete_mapping)
    decomposed_discrete_mapping[0] = sorted(values[2])
    decomposed_discrete_mapping.insert(0, sorted(values[1]))
    decomposed_discrete_mapping.insert(0, sorted(values[0]))
            
    queries=[]
    for i in range(100):
        query = []
        # for j in range(data_num_per_obj):
        for k in range(args.input_atoms):
            setting = np.random.randint(
                0,  (len(decomposed_discrete_mapping[k]))//2)
            query.append(setting)
        queries.append(query)
    print('queries', queries)
    np.save(os.path.join(args.save_folder, 'init_data.npy'), np.array(queries))
    

memory = Memory()
ppo = PPO(args, discrete_mapping, betas, eps_clip)

interval = 0.1
delta = False
noise_type = 'gaussian'
scenario = FrictionSliding(args.input_atoms, args.target_atoms,
                           interval, args.timesteps, delta, {'type': noise_type, 'value': None})
no_noise_train = np.load(
    'data/datasets/feat_train_causal_vel_sliding_spaced_new.npy')
scenario.add_noise['value'] = get_noise_std(
    args.input_atoms, no_noise_train, args.noise)
print('Noise variance', scenario.add_noise['value'])

simulator = RolloutSimulator(scenario)
learning_assess_values = [[0.5, 0.85], [0.5, 0.8], [
    0.5, 0.2, 0.1], [0.8, 1.1, 0.7], [0.5, 0.8], [0.5, 0.8]]
# pdb.set_trace()
learning_assess_data, learning_assess_data_noise = generate_dataset_discrete(
    learning_assess_values, scenario, True)
learning_assess_data = torch.from_numpy(learning_assess_data).float().cuda()

# env = AL_env(args, rel_rec, rel_send,
#              learning_assess_data, simulator, log_prior, logger, save_folder, valid_data_loader, valid_data.edge, train_data_min_max[0], train_data_min_max[1], discrete_mapping=discrete_mapping, discrete_mapping_grad=discrete_mapping_grad)
env = AL_env_entropy(args, rel_rec, rel_send,
             learning_assess_data, simulator, log_prior, logger, save_folder, valid_data_loader, valid_data.edge, train_data_min_max[0], train_data_min_max[1], discrete_mapping=discrete_mapping, discrete_mapping_grad=discrete_mapping_grad)

def main():
    # Train model
    best_val_loss = np.inf
    best_epoch = 0
    train_rl(env, memory, ppo)

    # print("Optimization Finished!")
    # print("Best Epoch: {:04d}".format(logger.best_epoch))
    # if args.save_folder:
    #     print("Best Epoch: {:04d}".format(logger.best_epoch), file=meta_file)
    #     meta_file.flush()

    # test_control(test_data_loader)
    # if meta_file is not None:
    #     print(save_folder)
    #     meta_file.close()


if __name__ == "__main__":
    main()
