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
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=700,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--control-batch-size', type=int, default=5,
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
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='_causal_vel_nohot',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
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
parser.add_argument('--skip-first', action='store_true', default=True,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=True,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')
parser.add_argument('--self-loop', action='store_true', default=False,
                    help='Whether graph contains self loop.')
parser.add_argument('--kl', type=float, default=1,
                    help='Whether to include kl as loss.')
parser.add_argument('--variations', type=int, default=5,
                    help='#values for one controlled var.')
parser.add_argument('--comment', type=str, default='',
                    help='Additional info for the run.')
parser.add_argument('--dataset_size', nargs='+',
                    help='#datapoints for train, val and test', required=True)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    args.val_writer = SummaryWriter(save_folder)
#     os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

#     pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

train_loader, valid_loader, test_loader = load_my_data(batch_size=args.batch_size, total_size=args.dataset_size,
                                                       suffix=args.suffix, self_loop=args.self_loop,
                                                       variations=args.variations, control_nodes=6, control=True,
                                                       control_batch_size=args.control_batch_size)

print(args, file=log)
log.flush()

if args.self_loop:
    # Assuming there is self loops
    off_diag = np.ones([args.num_atoms, args.num_atoms])
else:
    off_diag = np.ones([args.num_atoms, args.num_atoms]) - \
        np.eye(args.num_atoms)

# This is not adjacency matrix since it's 49*7, not 7*7!
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

if args.decoder == 'mlp':
    decoder = MLPDecoder_Causal(args)
elif args.decoder == 'rnn':
    decoder = RNNDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'sim':
    decoder = SimulationDecoder(
        loc_max, loc_min, vel_max, vel_min, args.suffix)

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))

    args.save_folder = False

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

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []

    rel_graphs = []
    rel_graphs_grad = []

    decoder.train()
    for batch_idx, (data, relations, which_node) in enumerate(train_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
            which_node = which_node.cuda()[0]
#         logits = encoder(data, rel_rec, rel_send)
#         edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)

        if args.decoder == 'rnn':
            output = decoder(data, edges, rel_rec, rel_send, 100,
                             burn_in=True,
                             burn_in_steps=args.timesteps - args.prediction_steps)
        else:
            output, logits = decoder(data, rel_rec, rel_send,
                                     args.temp, args.hard, args.prediction_steps)
        prob = my_softmax(logits, -1)

        # data: bs, #node, #timesteps, dim
        target = data[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, args.var)

        if args.prior:
            loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        else:
            loss_kl = kl_categorical_uniform(prob, args.num_atoms,
                                             args.edge_types)
        loss_kl *= args.kl
        loss = loss_nll + loss_kl
        if batch_idx < 5:
            a = nll_gaussian(output[:, -1, :, :],
                             target[:, -1, :, :], args.var)
            b = nll_gaussian(output[:, -2, :, :],
                             target[:, -2, :, :], args.var)
            c = nll_gaussian(output[:, -3, :, :],
                             target[:, -3, :, :], args.var)
            print(loss_nll, loss_kl, a, b, c)
        acc = edge_accuracy(logits, relations)
        acc_train.append(acc)

        optimizer.zero_grad()
        loss.backward()

        mask = torch.zeros(decoder.rel_graph.size(),
                           requires_grad=False, device="cuda")
        mask[:, :, (5-which_node):args.num_atoms**2:args.num_atoms] = 1
        mask[:, :, 6:args.num_atoms**2:args.num_atoms] = 1
        mask[:, :, 7:args.num_atoms**2:args.num_atoms] = 1
        decoder.rel_graph.grad *= mask

        optimizer.step()
        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())

        rel_graphs.append(decoder.rel_graph.detach().cpu().numpy())
        rel_graphs_grad.append(decoder.rel_graph.grad.detach().cpu().numpy())

    print(epoch, decoder.rel_graph.softmax(-1), decoder.rel_graph.size())
    if epoch % 10 == 0:
        torch.save(decoder.state_dict(), os.path.join(
            save_folder, str(epoch)+'_decoder.pt'))
        np.save(os.path.join(save_folder, str(epoch) +
                             '_rel_graph.npy'), np.array(rel_graphs))
        np.save(os.path.join(save_folder, str(epoch) +
                             '_rel_graph_grad.npy'), np.array(rel_graphs_grad))

    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []
    a_val = []
    b_val = []
    c_val = []

    decoder.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()

#         logits = encoder(data, rel_rec, rel_send)
#         edges = gumbel_softmax(logits, tau=args.temp, hard=True)
#

        # validation output uses teacher forcing
        output, logits = decoder(data, rel_rec, rel_send,
                                 args.temp, args.hard, 1)
        prob = my_softmax(logits, -1)

        target = data[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, args.var)
        a = nll_gaussian(output[:, -1, :, :], target[:, -1, :, :], args.var)
        b = nll_gaussian(output[:, -2, :, :], target[:, -2, :, :], args.var)
        c = nll_gaussian(output[:, -3, :, :], target[:, -3, :, :], args.var)

        if args.prior:
            loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        else:
            loss_kl = kl_categorical_uniform(prob, args.num_atoms,
                                             args.edge_types)
#         loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types, add_const=True)
#         acc = edge_accuracy(decoder.rel_graph.squeeze(), relations[0,-7:].unsqueeze(-1))
        acc = edge_accuracy(logits, relations)
        acc_val.append(acc)

        mse_val.append(F.mse_loss(output, target).item())
        nll_val.append(loss_nll.item())
        kl_val.append(loss_kl.item())
        a_val.append(a.item())
        b_val.append(b.item())
        c_val.append(c.item())

#     import pdb;pdb.set_trace()
#     print('Epoch: {:04d}'.format(epoch),
#           'nll_train: {:.10f}'.format(np.mean(nll_train)),
#           'kl_train: {:.10f}'.format(np.mean(kl_train)),
#           'mse_train: {:.10f}'.format(np.mean(mse_train)),
#           'acc_train: {:.10f}'.format(np.mean(acc_train)),
#           'nll_val: {:.10f}'.format(np.mean(nll_val)),
#           'a_val: {:.10f}'.format(np.mean(a_val)),
#           'b_val: {:.10f}'.format(np.mean(b_val)),
#           'c_val: {:.10f}'.format(np.mean(c_val)),
#           'kl_val: {:.10f}'.format(np.mean(kl_val)),
#           'mse_val: {:.10f}'.format(np.mean(mse_val)),
#           'acc_val: {:.10f}'.format(np.mean(acc_val)),
#           'time: {:.4f}s'.format(time.time() - t),
#           'lr: {:.6f}'.format(scheduler.get_lr()[0]), file=log)
    args.val_writer.add_scalar('nll_train', np.mean(nll_train), global_epoch)
    args.val_writer.add_scalar('kl_train', np.mean(kl_train), global_epoch)
    args.val_writer.add_scalar('mse_train', np.mean(mse_train), global_epoch)
    args.val_writer.add_scalar('acc_train', np.mean(acc_train), global_epoch)
    args.val_writer.add_scalar('nll_val', np.mean(nll_val), global_epoch)
    args.val_writer.add_scalar('-1_val', np.mean(a_val), global_epoch)
    args.val_writer.add_scalar('-2_val', np.mean(b_val), global_epoch)
    args.val_writer.add_scalar('-3_val', np.mean(c_val), global_epoch)
    args.val_writer.add_scalar('kl_val', np.mean(kl_val), global_epoch)
    args.val_writer.add_scalar('mse_val', np.mean(mse_val), global_epoch)
    args.val_writer.add_scalar('acc_val', np.mean(acc_val), global_epoch)
    args.val_writer.add_scalar('lr', scheduler.get_lr()[0], global_epoch)

    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save([decoder.state_dict(), decoder.rel_graph], decoder_file)
        print('Best model so far, saving...', file=log)
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
              'a_val: {:.10f}'.format(np.mean(a_val)),
              'b_val: {:.10f}'.format(np.mean(b_val)),
              'c_val: {:.10f}'.format(np.mean(c_val)),
              'kl_val: {:.10f}'.format(np.mean(kl_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(nll_val)


def test():
    acc_test = []
    nll_test = []
    kl_test = []
    mse_test = []
    tot_mse = 0
    counter = 0

    decoder.eval()
    decoder.load_state_dict(torch.load(decoder_file))
    for batch_idx, (data, relations) in enumerate(test_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()

#         assert (data.size(2) - args.timesteps) >= args.timesteps

#         data_encoder = data[:, :, :args.timesteps, :].contiguous()
        data_decoder = data[:, :, -args.timesteps:, :].contiguous()

#         logits = encoder(data_encoder, rel_rec, rel_send)
#         edges = gumbel_softmax(logits, tau=args.temp, hard=True)

        output, logits = decoder(data, rel_rec, rel_send,
                                 args.temp, args.hard, 1)
        prob = my_softmax(logits, -1)

        target = data_decoder[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, args.var)
        loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

        acc = edge_accuracy(logits, relations)
        acc_test.append(acc)

        mse_test.append(F.mse_loss(output, target).item())
        nll_test.append(loss_nll.item())
        kl_test.append(loss_kl.item())

        # For plotting purposes
        if args.decoder == 'rnn':
            if args.dynamic_graph:
                output = decoder(data, edges, rel_rec, rel_send, 100,
                                 burn_in=True, burn_in_steps=args.timesteps,
                                 dynamic_graph=True, encoder=encoder,
                                 temp=args.temp)
            else:
                output = decoder(data, edges, rel_rec, rel_send, 100,
                                 burn_in=True, burn_in_steps=args.timesteps)
            output = output[:, :, args.timesteps:, :]
            target = data[:, :, -args.timesteps:, :]
        else:
            data_plot = data[:, :, args.timesteps:args.timesteps + 21,
                             :].contiguous()
            output = decoder(data_plot, rel_rec, rel_send,
                             args.temp, args.hard, 20)
            target = data_plot[:, :, 1:, :]

        mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
        tot_mse += mse.data.cpu().numpy()
        counter += 1

    mean_mse = tot_mse / counter
    mse_str = '['
    for mse_step in mean_mse[:-1]:
        mse_str += " {:.12f} ,".format(mse_step)
    mse_str += " {:.12f} ".format(mean_mse[-1])
    mse_str += ']'

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(nll_test)),
          'kl_test: {:.10f}'.format(np.mean(kl_test)),
          'mse_test: {:.10f}'.format(np.mean(mse_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)))
    print('MSE: {}'.format(mse_str))
    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('nll_test: {:.10f}'.format(np.mean(nll_test)),
              'kl_test: {:.10f}'.format(np.mean(kl_test)),
              'mse_test: {:.10f}'.format(np.mean(mse_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)),
              file=log)
        print('MSE: {}'.format(mse_str), file=log)
        log.flush()


# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
global global_epoch
global_epoch = 0


for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    scheduler.step()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
    global_epoch += 1
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()

test()
if log is not None:
    print(save_folder)
    log.close()
