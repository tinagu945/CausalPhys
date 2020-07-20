"""Train data like Interpretable Physics"""
from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter


from utils import *
from models.modules_causal_vel import *

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=True,
                    help='Enables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=700,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--num-atoms', type=int, default=7,
                    help='Number of atoms in simulation.')
parser.add_argument('--suffix', type=str, default='_causal_vel_nohot',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=8,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=19,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=40,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--self-loop', action='store_true', default=False,
                    help='Whether graph contains self loop.')
parser.add_argument('--kl', type=float, default=1,
                    help='Whether to include kl as loss.')
parser.add_argument('--variations', type=int, default=5,
                    help='#values for one controlled var.')
parser.add_argument('--comment', type=str, default='',
                    help='Additional info for the run.')
parser.add_argument('--dataset_size', nargs='+', help='#datapoints for train, val and test', required=True)

args = parser.parse_args()
print(args)


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    meta_file = open(os.path.join((save_folder, 'meta.txt')),'r')
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")


print(args, file=meta_file)
meta_file.flush()

if args.self_loop:
    off_diag = np.ones([args.num_atoms, args.num_atoms])
else:
    off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

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

if args.prior:
    prior = np.array([0.9, 0.1])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)
    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()
    
    
    
    
        

def main():
    # Train model
    best_val_loss = np.inf
    best_epoch = 0
    global global_epoch
    global_epoch=0
    trajectory_len=19
    
    _, valid_loader, test_loader = load_AL_data(batch_size=args.batch_size, total_size=args.dataset_size,\
                                                       suffix=args.suffix, self_loop=args.self_loop, \
                                                       variations=args.variations, control_nodes=6, control=True)
    func =
    simulator = ControlOracle(func, trajectory_len, args.num_nodes, low=0, high=100, \
                 control_low=20, control_high=50)
    uncertain_sampler = MaximalEntropySimulatorSampler(simulator)
    random_sampler = RandomSimulatorSampler(simulator)
    
    for epoch in range(args.epochs):
        new_data_idx, uncertain_nodes = sampler.sample(decoder.rel_graph)
        train_loader = update_AL_dataset(new_data_idx, uncertain_nodes, train_dataset)   
    
        val_loss = train(train_loader, valid_loader, epoch, best_val_loss)
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        global_epoch +=1
        
        
        
        
        
    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch))
    if args.save_folder:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

    test(test_loader)
    if log is not None:
        print(save_folder)
        log.close()
        

        
if __name__ == "__main__":
    main()
