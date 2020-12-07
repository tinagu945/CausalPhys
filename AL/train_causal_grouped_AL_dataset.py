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
from data.AL_sampler import *
from data.datasets import *
# from data.dataset_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--train-bs', type=int, default=144,
                    help='Number of samples per batch during training.')
parser.add_argument('--val-bs', type=int, default=128,
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
parser.add_argument('--save-folder', type=str, default='logs_AL',
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
                    help='Additional info for the run to be recorded in logs.')
parser.add_argument('--sampler', type=str, default='uncertainty',
                    help='Sample method.')
parser.add_argument('--train-size', type=int, default=None,
                    help='#datapoints for train')
parser.add_argument('--val-size', type=int, default=None,
                    help='#datapoints for val')
parser.add_argument('--test-size', type=int, default=None,
                    help='#datapoints for test')
parser.add_argument('--grouped', action='store_true', default=True,
                    help='Whether we want to do the grouped training. Should always be true for this script.')
parser.add_argument('--need-grouping', action='store_true', default=False,
                    help='If grouped is True, whether the dataset actually needs grouping.')
parser.add_argument('--val-need-grouping', action='store_true', default=True,
                    help='If grouped is True, whether the validation dataset actually needs grouping.')
parser.add_argument('--val-grouped', action='store_true', default=True,
                    help='Whether to group the valid and test dataset')
parser.add_argument('--control-constraint', type=float, default=0.0,
                    help='Coefficient for control constraint loss')
parser.add_argument('--gt-A', action='store_true', default=False,
                    help='Whether use the ground truth adjacency matrix, useful for debuging the encoder.')
parser.add_argument('--train-log-freq', type=int, default=20,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--val-log-freq', type=int, default=10,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--log-data-size', type=int, default=300,
                    help='See how much data as one epoch.')
parser.add_argument('--all-connect', action='store_true', default=False,
                    help='Whether the adjancency matrix is fully connected and not trainable.')
parser.add_argument('--sample-num-groups', type=int, default=1,
                    help='# groups for each sampling.')
parser.add_argument('--budget', type=int, default=46656,
                    help='# data allowed before termination.')
parser.add_argument('--topk', type=int, default=1,
                    help='Top k nodes with highest criterion score to be queried.')


args = parser.parse_args()
args.num_atoms = args.input_atoms+args.target_atoms
args.script = 'train_causal_grouped'
if args.gt_A or args.all_connect:
    print('Using given graph and kl loss will be omitted')
    args.kl = 0
assert (args.variations*args.sample_num_groups) % args.log_data_size == 0, "Please ensure (args.variations*args.sample_num_groups) can be divided by args.log_data_size, else epoch number may not be integer."
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
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

decoder = MLPDecoder_Causal(args)
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
log_prior = torch.FloatTensor(np.log(prior))
log_prior = torch.unsqueeze(log_prior, 0)
log_prior = torch.unsqueeze(log_prior, 0)

log_prior = log_prior.cuda()
decoder.cuda()
rel_rec = rel_rec.cuda()
rel_send = rel_send.cuda()
triu_indices = triu_indices.cuda()
tril_indices = tril_indices.cuda()


def main():
    # Train model
    best_val_loss = np.inf
    best_epoch = 0

    # AL grouped must use controlled training dataset
    train_dataset = OneGraphDataset.load_one_graph_data(
        'train_causal_vel_'+args.suffix, train_data_min_max=None, size=args.train_size, self_loop=args.self_loop, control=args.grouped, control_nodes=args.input_atoms, variations=args.variations, need_grouping=args.need_grouping)

    if args.val_grouped:
        # To see control loss, val and test should be grouped
        valid_dataset = OneGraphDataset.load_one_graph_data(
            'valid_causal_vel_'+args.val_suffix, train_data_min_max=[train_dataset.mins, train_dataset.maxs], size=args.val_size, self_loop=args.self_loop, control=True, control_nodes=args.input_atoms, variations=args.val_variations, need_grouping=args.val_need_grouping)
        valid_sampler = RandomPytorchSampler(valid_dataset)
        valid_data_loader = DataLoader(
            valid_dataset, batch_size=args.val_bs, shuffle=False, sampler=valid_sampler)

        # test_data = load_one_graph_data(
        #     'test_'+args.val_suffix, size=args.test_size, self_loop=args.self_loop, control=True, control_nodes=args.input_atoms, variations=4)
        # test_sampler = RandomPytorchSampler(test_data)
        # test_data_loader = DataLoader(
        #     test_data, batch_size=args.val_bs, shuffle=False, sampler=test_sampler)
    else:
        valid_dataset = OneGraphDataset.load_one_graph_data(
            'valid_causal_vel_'+args.val_suffix, train_data_min_max=[train_dataset.mins, train_dataset.maxs], size=args.val_size, self_loop=args.self_loop, control=False)
        valid_data_loader = DataLoader(
            valid_dataset, batch_size=args.val_bs, shuffle=True)
        # test_data = load_one_graph_data(
        #     'test_'+args.val_suffix, size=args.val_size, self_loop=args.self_loop, control=False)
        # test_data_loader = DataLoader(
        #     test_data, batch_size=args.val_bs, shuffle=True)

    if args.sampler == 'random':
        sampler = RandomDatasetSampler(train_dataset, args)
    elif args.sampler == 'uncertainty':
        sampler = MaximalEntropyDatasetSampler(train_dataset, args)
    else:
        print('Only random and uncertainty samplers are supported for now!')

    logger = Logger(save_folder)

    print('Doing initial validation before training...')
    val_control(
        args, log_prior, logger, save_folder, valid_data_loader, -1, decoder, rel_rec, rel_send, scheduler)

    data_idx = []
    nodes = []
    # Start with one group of each node
    for i in range(args.input_atoms):
        # group_size=args.variations
        new_data_idx = sampler.sample([i], args.variations, 200)
        uncertain_nodes = torch.LongTensor([i]*args.variations*200).cuda()
        data_idx.append(new_data_idx)
        nodes.append(uncertain_nodes)

    data_idx = torch.cat(data_idx)
    nodes = torch.cat(nodes)
    al_train_dataset = ALIndexDataset(train_dataset, data_idx, nodes)

    # while len(al_train_dataset.idxs) < args.budget:
    for epoch in range(500):
        # epoch = int(len(al_train_dataset)/args.log_data_size)
        # Sampler batch_size is fixed to be #value variations for a node
        # control_nodes = sampler.criterion(decoder.rel_graph, k=args.topk)
        # new_data_idx = sampler.sample(
        #     control_nodes, args.variations, args.sample_num_groups)
        # uncertain_nodes = torch.LongTensor(
        #     control_nodes).repeat_interleave(args.variations*args.sample_num_groups).cuda()
        # al_train_dataset.update(new_data_idx, uncertain_nodes)

        # with open(os.path.join(save_folder, str(epoch)+'_queries.pt'), 'wb') as f:
        #     torch.save(
        #         [al_train_dataset.nodes, al_train_dataset.idxs], f)
        #     print('sampled nodes this episode',
        #           control_nodes, new_data_idx, len(al_train_dataset))

        # Only need to be a normal dataloader since the grouping and indexing are already implemented in al_train_dataset. Just make sure shuffle=False.
        train_data_loader = DataLoader(
            al_train_dataset, batch_size=args.train_bs, shuffle=False)

        # for j in range(10):
        # print('epoch and j', epoch, j)
        nll, nll_lasttwo, kl, mse, control_constraint_loss, lr, rel_graphs, rel_graphs_grad, a, b, c, d, e, f = train_control(
            args, log_prior, optimizer, save_folder, train_data_loader, valid_data_loader, decoder, rel_rec, rel_send, epoch)

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
