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
from train import train_control
from val import val_control
from test import test_control
from utils.logger import Logger
from AL.AL_control_sampler import RandomPytorchSampler
from data.datasets import *
from data.dataset_utils import *
from utils.general_parser import general_parser

parser = general_parser()
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--grouped', action='store_true', default=False,
                    help='Whether we want to do the grouped training.')
parser.add_argument('--need-grouping', action='store_true', default=False,
                    help='If grouped is True, whether the dataset actually needs grouping.')

args = parser.parse_args()
args.num_atoms = args.input_atoms+args.target_atoms
args.script = 'train_causal_grouped'
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

# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(args.num_atoms)
tril_indices = get_tril_offdiag_indices(args.num_atoms)

prior = np.array([0.9, 0.1])  # TODO: hard coded for now
print("Using prior")
print(prior)
log_prior = torch.FloatTensor(np.log(prior)).cuda()
log_prior = torch.unsqueeze(log_prior, 0).cuda()
log_prior = torch.unsqueeze(log_prior, 0).cuda()


def main():
    # Train model
    best_val_loss = np.inf
    best_epoch = 0

    if args.grouped:
        assert args.train_bs % args.variations == 0, "Grouping training set requires args.traing-bs integer times of args.variations"

        train_data = OneGraphDataset.load_one_graph_data(
            'train_causal_vel_'+args.suffix, train_data_min_max=None, size=args.train_size, self_loop=args.self_loop, control=args.grouped, control_nodes=args.input_atoms, variations=args.variations, need_grouping=args.need_grouping)
        train_sampler = RandomPytorchSampler(train_data)
        train_data_loader = DataLoader(
            train_data, batch_size=args.train_bs, shuffle=False, sampler=train_sampler)

    else:
        train_data = OneGraphDataset.load_one_graph_data(
            'train_causal_vel_'+args.suffix, train_data_min_max=None, size=args.train_size, self_loop=args.self_loop, control=args.grouped)
        train_data_loader = DataLoader(
            train_data, batch_size=args.train_bs, shuffle=True)

    if args.val_grouped:
        # To see control loss, val and test should be grouped
        valid_data = OneGraphDataset.load_one_graph_data(
            'valid_causal_vel_'+args.val_suffix, train_data_min_max=[train_data.mins, train_data.maxs], size=args.val_size, self_loop=args.self_loop, control=True, control_nodes=args.input_atoms, variations=args.val_variations, need_grouping=args.val_need_grouping)
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
            'valid_causal_vel_'+args.val_suffix, train_data_min_max=[train_data.mins, train_data.maxs], size=args.val_size, self_loop=args.self_loop, control=False)
        valid_data_loader = DataLoader(
            valid_data, batch_size=args.val_bs, shuffle=True)
        # test_data = load_one_graph_data(
        #     'test_'+args.val_suffix, size=args.val_size, self_loop=args.self_loop, control=False)
        # test_data_loader = DataLoader(
        #     test_data, batch_size=args.val_bs, shuffle=True)
    print('size of training dataset', len(train_data),
          'size of valid dataset', len(valid_data))
    logger = Logger(save_folder)
    print('Doing initial validation before training...')
    nll_val, nll_lasttwo_val, kl_val, mse_val, a_val, b_val, c_val, control_constraint_loss_val, nll_lasttwo_5_val, nll_lasttwo_10_val, nll_lasttwo__1_val, nll_lasttwo_1_val = val_control(
        args, log_prior, logger, args.save_folder, valid_data_loader, -1, decoder)

    for epoch in range(args.epochs):
        # TODO: when len(train_dataset) reaches budget, force stop
        # print('#batches in train_dataset', len(train_dataset)/args.train_bs)
        nll, nll_lasttwo, kl, mse, control_constraint_loss, lr, rel_graphs, rel_graphs_grad, a, b, c, d, e = train_control(
            args, log_prior, optimizer, save_folder, train_data_loader, decoder, epoch)

        if epoch % args.train_log_freq == 0:
            logger.log('train', decoder, epoch, nll, nll_lasttwo, kl=kl, mse=mse, control_constraint_loss=control_constraint_loss, lr_train=lr, rel_graphs=rel_graphs,
                       rel_graphs_grad=rel_graphs_grad, msg_hook_weights=a, nll_train_lasttwo=b, nll_lasttwo_10_train=c, nll_lasttwo__1_train=d, nll_lasttwo_1_train=e)

        if epoch % args.train_log_freq == 0:
            nll_val, nll_lasttwo_val, kl_val, mse_val, a_val, b_val, c_val, control_constraint_loss_val, nll_lasttwo_5_val, nll_lasttwo_10_val, nll_lasttwo__1_val, nll_lasttwo_1_val = val_control(
                args, log_prior, logger, args.save_folder, valid_data_loader, epoch, decoder)

            logger.log('val', decoder, epoch, nll_val, nll_lasttwo_val, kl_val=kl_val, mse_val=mse_val, a_val=a_val, b_val=b_val, c_val=c_val, control_constraint_loss_val=control_constraint_loss_val,
                       nll_lasttwo_5_val=nll_lasttwo_5_val,  nll_lasttwo_10_val=nll_lasttwo_10_val, nll_lasttwo__1_val=nll_lasttwo__1_val, nll_lasttwo_1_val=nll_lasttwo_1_val, scheduler=scheduler)

        # if epoch % args.val_log_freq == 0:
        #     _ = val_control(
        #         args, log_prior, logger, save_folder, valid_data_loader, epoch, decoder, scheduler)
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
