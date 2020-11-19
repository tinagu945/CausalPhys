"""Train data like Interpretable Physics"""
import time
import argparse
import os
import datetime
import sys

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.functions import *
from models.modules_causal_vel import *
from RL.train_rl import train_rl
# from  import val_control
# from test import test_control
from utils.logger import Logger
from data.AL_sampler import RandomPytorchSampler
from data.datasets import *
from data.dataset_utils import *
from RL.PPO_discrete import *
from AL_env import *
from data.simulator import RolloutSimulator
from data.scenarios import FrictionSliding

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--rl_epochs', type=int, default=500,
                    help='Number of epochs to train the rl policy inside each epoch of learning.')
parser.add_argument('--train-bs', type=int, default=10,
                    help='Number of samples per batch during training.')
parser.add_argument('--val-bs', type=int, default=10,
                    help='Number of samples per batch during validation and test.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Initial learning rate.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--input-atoms', type=int, default=6,
                    help='Number of atoms need to be controlled in simulation.')
parser.add_argument('--suffix', type=str, default='causal_vel_delta_grouped_46656',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--val-suffix', type=str, default=None,
                    help='Suffix for valid and testing data (e.g. "_charged".')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Probability of an element to be zeroed.')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model and logs.')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=9,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=40,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=20, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=40,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=True,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--self-loop', action='store_true', default=True,
                    help='Whether graph contains self loop.')
parser.add_argument('--kl', type=float, default=10,
                    help='Whether to include kl as loss.')
parser.add_argument('--variations', type=int, default=6,
                    help='#values for one controlled var in training dataset.')
parser.add_argument('--val-variations', type=int, default=4,
                    help='#values for one controlled var in validation dataset.')
parser.add_argument('--target-atoms', type=int, default=2,
                    help='#atoms for results.')
parser.add_argument('--comment', type=str, default='',
                    help='Additional info for the run.')
parser.add_argument('--train-size', type=int, default=None,
                    help='#datapoints for train')
parser.add_argument('--val-size', type=int, default=None,
                    help='#datapoints for val')
parser.add_argument('--test-size', type=int, default=None,
                    help='#datapoints for test')
parser.add_argument('--val-need-grouping', action='store_true', default=True,
                    help='If grouped is True, whether the validation dataset actually needs grouping.')
parser.add_argument('--val-grouped', action='store_true', default=True,
                    help='Whether to group the valid and test dataset')
parser.add_argument('--control-constraint', type=float, default=0.0,
                    help='Coefficient for control constraint loss')
parser.add_argument('--gt-A', action='store_true', default=False,
                    help='Whether use the ground truth adjacency matrix, useful for debuging the encoder.')
parser.add_argument('--train-log-freq', type=int, default=10,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--val-log-freq', type=int, default=5,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--all-connect', action='store_true', default=False,
                    help='Whether the adjancency matrix is fully connected and not trainable.')

parser.add_argument('--rl-lr', type=float, default=0.002,
                    help='lr to train the rl policy')
parser.add_argument('--rl-gamma', type=float, default=0.99,
                    help='discount factor of the rl policy')
parser.add_argument('--rl-hidden', type=int, default=64,
                    help='Number of hidden units for the policy network.')
parser.add_argument('--extract-feat-dim', type=int, default=64,
                    help='Hidden dimension for obj_extractor, obj_data_extractor, learning_extractor and learning_assess_extractor.')
parser.add_argument('--budget', type=int, default=100,
                    help='If the causal model queried for more data than budget, env reset.')
parser.add_argument('--initial_obj_num', type=int, default=6,
                    help='State dimension of the MDP.')
# TODO:
parser.add_argument('--noise', action='store_true', default=False,
                    help='Whether the simulator adds noise to training data.')

args = parser.parse_args()
assert args.train_bs == args.val_bs
args.num_atoms = args.input_atoms+args.target_atoms
args.script = 'train_causal_grouped'
args.state_dim = 5*args.extract_feat_dim
if args.gt_A or args.all_connect:
    print('Using given graph and kl loss will be omitted')
    args.kl = 0

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Save model and meta-data. Always saves in a new sub-folder.
now = datetime.datetime.now()
timestamp = now.isoformat()
save_folder = '{}/exp{}/'.format(args.save_folder,
                                 '_'.join([timestamp]+[i.replace("--", "") for i in sys.argv[1:]]))
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

decoder = MLPDecoder_Causal(args, rel_rec, rel_send).cuda()
optimizer = optim.Adam(list(decoder.parameters())+[decoder.rel_graph],
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

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
memory = Memory()
ppo = PPO(args.state_dim, args.variations, args.rl_hidden, args.rl_lr,
          betas, args.rl_gamma, args.rl_epochs, eps_clip, 4)

discrete_mapping_grad = [[0.53, 0.637, 0.745, 1.284, 1.176, 1.5], [
    0, 0.11, 0.44, 0.77, 0.88, 1], [0, 0.11, 0.33, 0.44, 0.55, 1]]
discrete_mapping = [lambda x: 0.53+x *
                    (1.5-0.53)/5, lambda x: 0+x*(1-0)/5, lambda x: 0+x*(1-0)/5]
learning_assess_data = torch.zeros((20, 8, 40, 9)).cuda()
interval = 0.1
delta = False
scenario = FrictionSliding(args.input_atoms, args.target_atoms,
                           interval, args.timesteps, delta, args.noise)
simulator = RolloutSimulator(scenario)
env = AL_env(args, decoder, optimizer, scheduler,
             learning_assess_data, simulator, log_prior, logger, save_folder, valid_data_loader, valid_data.edge, train_data_min_max[0], train_data_min_max[1], discrete_mapping=discrete_mapping, discrete_mapping_grad=discrete_mapping_grad)
# shape_color_mu=[[0,1,0.33,0.88,0.22,0.77], [0,1,0.44,0.55,0.33,0.88], [0,0.56,0.497,0.249,0.373, 0.06]]
env.obj = {0: [0, 0, 0], 1: [1, 1, 0.56], 2: [0.33, 0.44, 0.497], 3: [
    0.88, 0.55, 0.249], 4: [0.22, 0.33, 0.373], 5: [0.77, 0.88, 0.06]}
# env.init_train_data(data_num_per_obj=1)


def main():
    # Train model
    best_val_loss = np.inf
    best_epoch = 0
    print('Doing initial validation before training...')
    # val_rl(
    #     args, log_prior, logger, save_folder, valid_data_loader, -1, decoder, rel_rec, rel_send, scheduler)

    for epoch in range(args.epochs):
        nll, nll_lasttwo, kl, mse, control_constraint_loss, lr, rel_graphs, rel_graphs_grad, a, b, c, d, e, f = train_rl(
            args, env, memory, ppo)

        if epoch % args.train_log_freq == 0:
            logger.log('train', decoder, epoch, nll, nll_lasttwo, kl=kl, mse=mse, control_constraint_loss=control_constraint_loss, lr=lr, rel_graphs=rel_graphs,
                       rel_graphs_grad=rel_graphs_grad, msg_hook_weights=a, nll_train_lasttwo=b, nll_train_lasttwo_5=c, nll_train_lasttwo_10=d, nll_train_lasttwo__1=e, nll_train_lasttwo_1=f)

        if epoch % args.val_log_freq == 0:
            _ = val_control(
                args, log_prior, logger, save_folder, valid_data_loader, epoch, decoder, rel_rec, rel_send, scheduler)
        scheduler.step()

    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(logger.best_epoch))
    if args.save_folder:
        print("Best Epoch: {:04d}".format(logger.best_epoch), file=meta_file)
        meta_file.flush()

    test_control(test_data_loader)
    if meta_file is not None:
        print(save_folder)
        meta_file.close()


if __name__ == "__main__":
    main()
