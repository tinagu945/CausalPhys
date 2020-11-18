import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import os
from models.modules_causal_vel import *
from data.AL_sampler import RandomPytorchSampler
from data.datasets import *
from data.dataset_utils import *
import argparse
from torch.utils.data import DataLoader
from utils.functions import *


def load_predict(args, weight_path, start_ind=0, stop_ind=5, record=True):
    weight_path = 'logs/'+weight_path
    decoder = MLPDecoder_Causal(args).cuda()
    off_diag = np.ones([args.num_atoms, args.num_atoms])
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec).cuda()
    rel_send = torch.FloatTensor(rel_send).cuda()

    train_data = OneGraphDataset.load_one_graph_data(
        'train_causal_vel_'+args.suffix, train_data=None, size=None, self_loop=args.self_loop, control=False)
    if args.val_grouped:
        # To see control loss, val and test should be grouped
        valid_data = OneGraphDataset.load_one_graph_data(
            'valid_causal_vel_'+args.val_suffix, train_data=train_data, size=args.val_size, self_loop=args.self_loop, control=True, control_nodes=args.input_atoms, variations=args.val_variations, need_grouping=args.val_need_grouping)
        valid_sampler = RandomPytorchSampler(valid_data)
        valid_data_loader = DataLoader(
            valid_data, batch_size=args.val_bs, shuffle=False, sampler=valid_sampler)
    else:
        valid_data = OneGraphDataset.load_one_graph_data(
            'valid_causal_vel_'+args.val_suffix, train_data=train_data, size=args.val_size, self_loop=args.self_loop, control=False)
        valid_data_loader = DataLoader(
            valid_data, batch_size=args.val_bs, shuffle=False)

    decoder.load_state_dict(torch.load(weight_path)[0])
    decoder.eval()
    # if not args.gt_A:
    # w = 'logs/exp2020-09-15T17:49:41.582904_suffix_SHO_cont_nowall_40_new_val-suffix_SHO_spaced_nowall_40_new_interpolation_val-grouped_val-need-grouping_grouped_control-constraint_2.0_gt-A_decoder-hidden_256/best_decoder.pt'

    # w = 'logs/exp2020-09-16T16:29:53.485070_suffix_airfall_cont_40_new_val-suffix_airfall_spaced_40_new_interpolation_val-grouped_val-need-grouping_grouped_control-constraint_2.0_gt-A/best_decoder.pt'
    # print('Loading rel_graph from', w)
    # decoder.rel_graph = torch.load(w)[1].cuda()
    decoder.rel_graph = torch.load(weight_path)[1].cuda()
    # decoder.rel_graph[:, :, :-16, 1] = 10

    # decoder.rel_graph = torch.zeros([1, 1, 64, 2]).cuda()
    # # for i in [59, 48, 52, 54, 62]:
    # decoder.rel_graph[:, :, -16:, 1] = 100
    # decoder.rel_graph[:, :, :, 0] = 10-decoder.rel_graph[:, :, :, 1]
    # print(decoder.rel_graph.softmax(-1))
    # edge_acc = edge_accuracy(decoder.rel_graph, train_data.edge)
    # print('edge accuracy at the best epoch', edge_acc)
    loss = []
    truth = []
    pred = []
    condition = []
    for batch_idx, all_data in enumerate(valid_data_loader):
        if batch_idx < start_ind:
            pass
        if batch_idx < stop_ind and (batch_idx > start_ind or batch_idx == start_ind):
            if args.val_grouped:
                # edge is only for calculating edge accuracy. Since we have not included that, edge is not used.
                data, which_node, edge = all_data[0].cuda(
                ), all_data[1].cuda(), all_data[2].cuda()
                output, logits, msg_hook = decoder(data, rel_rec, rel_send,
                                                   args.temp, args.hard, args.prediction_steps, [])
                control_constraint_loss = control_loss(
                    msg_hook, which_node, args.input_atoms, args.variations)*args.control_constraint

            else:
                data, edge = all_data[0].cuda(), all_data[1].cuda()
                output, logits, msg_hook = decoder(data, rel_rec, rel_send,
                                                   args.temp, args.hard, args.prediction_steps, [])
                control_constraint_loss = torch.zeros(1).cuda()
            # print('batch_size', data.size(0))
            target = data[:, :, 1:, :]
            loss_nll, _ = nll_gaussian(
                output[:, -2:, :, :], target[:, -2:, :, :], args.var)
            loss.append(loss_nll.item())

            if record:
                print('Nll', loss_nll)
                target, output = denormalize(target, output, train_data)
                print('Setup [shapes,colors,mus,thetas,masses,v0s]',
                      batch_idx, target[0, :-2, 0, 0])
                print('Velocity', batch_idx,
                      target[0, -2, :, 0], '\n', output[0, -2, :, 0])
                print('Position', batch_idx,
                      target[0, -1, :, 0], '\n', output[0, -1, :, 0])
                condition.append(target[:, :-2, 0, 0])
                truth.append(target)
                pred.append(output)
        elif batch_idx > stop_ind:
            break

    print('Avg nll loss', np.mean(loss), weight_path)
    return loss, truth, pred, condition, valid_data


def denormalize(target, output, train_data):
    for i in range(target.size(1)):
        output[:, i, :, 0] = (
            (output[:, i, :, 0]+1)*(train_data.maxs[i]-train_data.mins[i]))/2+train_data.mins[i]
        target[:, i, :, 0] = (
            (target[:, i, :, 0]+1)*(train_data.maxs[i]-train_data.mins[i]))/2+train_data.mins[i]
    return target, output


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
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
    parser.add_argument('--hard', action='store_true', default=True,
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
    parser.add_argument('--val-size', type=int, default=4096,
                        help='#datapoints for val')
    parser.add_argument('--test-size', type=int, default=None,
                        help='#datapoints for test')
    parser.add_argument('--grouped', action='store_true', default=False,
                        help='Whether we want to do the grouped training.')
    parser.add_argument('--need-grouping', action='store_true', default=False,
                        help='If grouped is True, whether the dataset actually needs grouping.')
    parser.add_argument('--val-need-grouping', action='store_true', default=False,
                        help='If grouped is True, whether the validation dataset actually needs grouping.')
    parser.add_argument('--val-grouped', action='store_true', default=False,
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
    parser.add_argument('--weight-path', type=str, default='',
                        help='The number of input dimensions (position + velocity).')

    args = parser.parse_args()
    args.num_atoms = args.input_atoms+args.target_atoms
    args.script = 'train_causal_grouped'

    sliding_path = ['exp2020-09-29T13:00:40.339006_suffix_sliding_cont_fixedgaussian0.1_new_val-suffix_sliding_spaced_fixedgaussian0.1_new_interpolation_all-connect',
                    'exp2020-09-29T13:01:34.100094_suffix_sliding_cont_fixedgaussian0.1_new_val-suffix_sliding_spaced_fixedgaussian0.1_new_interpolation', 'exp2020-09-29T13:02:04.713846_suffix_sliding_cont_fixedgaussian0.1_new_val-suffix_sliding_spaced_fixedgaussian0.1_new_interpolation_gt-A', 'exp2020-09-29T13:04:44.772277_suffix_sliding_cont_fixedgaussian0.1_new_val-suffix_sliding_spaced_fixedgaussian0.1_new_interpolation_gt-A_seed_1', 'exp2020-09-29T13:06:09.740529_suffix_sliding_cont_fixedgaussian0.1_new_val-suffix_sliding_spaced_fixedgaussian0.1_new_interpolation_seed_1', 'exp2020-09-29T13:07:16.690882_suffix_sliding_cont_fixedgaussian0.1_new_val-suffix_sliding_spaced_fixedgaussian0.1_new_interpolation_all-connect_seed_1', 'exp2020-10-01T02:24:35.245564_suffix_sliding_cont_fixedgaussian0.1_new_val-suffix_sliding_spaced_fixedgaussian0.1_new_interpolation_seed_2', 'exp2020-10-01T02:30:47.609654_suffix_sliding_cont_fixedgaussian0.1_new_val-suffix_sliding_spaced_fixedgaussian0.1_new_interpolation_all-connect_seed_2']

    airfall_path = ['exp2020-09-28T17:55:12.546640_suffix_airfall_cont_fixedgaussian0.1_new_val-suffix_airfall_spaced_fixedgaussian0.1_new_interpolation_all-connect',
                    'exp2020-09-28T17:53:51.266094_suffix_airfall_cont_fixedgaussian0.1_new_val-suffix_airfall_spaced_fixedgaussian0.1_new_interpolation', 'exp2020-09-28T17:52:19.705827_suffix_airfall_cont_fixedgaussian0.1_new_val-suffix_airfall_spaced_fixedgaussian0.1_new_interpolation_gt-A', 'exp2020-10-01T15:45:26.180443_suffix_airfall_cont_fixedgaussian0.1_new_val-suffix_airfall_spaced_fixedgaussian0.1_new_interpolation_all-connect_seed_1', 'exp2020-10-01T15:46:27.843718_suffix_airfall_cont_fixedgaussian0.1_new_val-suffix_airfall_spaced_fixedgaussian0.1_new_interpolation_seed_1', 'exp2020-10-02T03:12:47.318536_suffix_airfall_cont_fixedgaussian0.1_new_val-suffix_airfall_spaced_fixedgaussian0.1_new_interpolation_gt-A_seed_1', 'exp2020-10-02T03:13:11.391255_suffix_airfall_cont_fixedgaussian0.1_new_val-suffix_airfall_spaced_fixedgaussian0.1_new_interpolation_gt-A_seed_2']

    airfall_path_group = ['exp2020-09-28T17:56:23.373702_suffix_airfall_cont_fixedgaussian0.1_new_val-suffix_airfall_spaced_fixedgaussian0.1_new_interpolation_all-connect_grouped_control-constraint_2.0',
                          'exp2020-09-28T17:58:15.205180_suffix_airfall_cont_fixedgaussian0.1_new_val-suffix_airfall_spaced_fixedgaussian0.1_new_interpolation_grouped_control-constraint_2.0', 'exp2020-09-28T18:00:49.162823_suffix_airfall_cont_fixedgaussian0.1_new_val-suffix_airfall_spaced_fixedgaussian0.1_new_interpolation_grouped_control-constraint_2.0_gt-A']

    SHO_path = ['exp2020-09-28T01:54:36.339337_suffix_SHO_cont_fixedgaussian0.1_new_val-suffix_SHO_spaced_fixedgaussian0.1_new_interpolation_gt-A', 'exp2020-09-28T01:52:38.600905_suffix_SHO_cont_fixedgaussian0.1_new_val-suffix_SHO_spaced_fixedgaussian0.1_new_interpolation_all-connect',
                'exp2020-09-28T01:50:16.683673_suffix_SHO_cont_fixedgaussian0.1_new_val-suffix_SHO_spaced_fixedgaussian0.1_new_interpolation', 'exp2020-10-01T02:34:08.437066_suffix_SHO_cont_fixedgaussian0.1_new_val-suffix_SHO_spaced_fixedgaussian0.1_new_interpolation_all-connect_seed_2', 'exp2020-10-01T02:41:00.624098_suffix_SHO_cont_fixedgaussian0.1_new_val-suffix_SHO_spaced_fixedgaussian0.1_new_interpolation_gt-A_seed_2', 'exp2020-10-01T15:28:53.840221_suffix_SHO_cont_fixedgaussian0.1_new_val-suffix_SHO_spaced_fixedgaussian0.1_new_interpolation_seed_2', 'exp2020-10-01T15:29:54.618608_suffix_SHO_cont_fixedgaussian0.1_new_val-suffix_SHO_spaced_fixedgaussian0.1_new_interpolation_all-connect_seed_3', 'exp2020-10-01T15:33:32.392524_suffix_SHO_cont_fixedgaussian0.1_new_val-suffix_SHO_spaced_fixedgaussian0.1_new_interpolation_gt-A_seed_3', 'exp2020-10-02T02:59:38.868375_suffix_SHO_cont_fixedgaussian0.1_new_val-suffix_SHO_spaced_fixedgaussian0.1_new_interpolation_seed_3']

    sliding_path_group = ['exp2020-09-22T15:53:48.129713_suffix_sliding_cont_40_fixedgaussian05_new_val-suffix_sliding_spaced_40_fixedgaussian05_new_interpolation_val-grouped_val-need-grouping_grouped_control-constraint_2.0_gt-A',
                          'exp2020-09-22T15:55:26.763593_suffix_sliding_cont_40_fixedgaussian05_new_val-suffix_sliding_spaced_40_fixedgaussian05_new_interpolation_val-grouped_val-need-grouping_grouped_control-constraint_2.0_all-connect', 'exp2020-09-23T03:24:07.427021_suffix_sliding_cont_40_fixedgaussian05_new_val-suffix_sliding_spaced_40_fixedgaussian05_new_interpolation_val-grouped_val-need-grouping_grouped_control-constraint_2.0']

    SHO_path_group = ['exp2020-09-28T01:55:51.057911_suffix_SHO_cont_fixedgaussian0.1_new_val-suffix_SHO_spaced_fixedgaussian0.1_new_interpolation_grouped_control-constraint_2.0',
                      'exp2020-09-28T01:57:21.920209_suffix_SHO_cont_fixedgaussian0.1_new_val-suffix_SHO_spaced_fixedgaussian0.1_new_interpolation_grouped_control-constraint_2.0_gt-A', 'exp2020-09-28T01:58:52.226227_suffix_SHO_cont_fixedgaussian0.1_new_val-suffix_SHO_spaced_fixedgaussian0.1_new_interpolation_grouped_control-constraint_2.0_all-connect']

    SHO_suffix = 'SHO_cont_fixedgaussian0.1_new'
    SHO_val_suffix = 'SHO_spaced_fixedgaussian0.1_new_interpolation'
    # SHO_val_suffix = 'SHO_spaced_fixedgaussian0.1_new_extrapolation_0_0.2'
    airfall_suffix = 'airfall_cont_fixedgaussian0.1_new'
    # airfall_val_suffix = 'airfall_spaced_fixedgaussian0.1_new_interpolation'
    airfall_val_suffix = 'airfall_spaced_fixedgaussian0.1_new_extrapolation_0.4_0.6'
    sliding_suffix = 'sliding_cont_fixedgaussian0.1_new'
    # sliding_val_suffix = 'sliding_spaced_fixedgaussian0.1_new_interpolation'
    sliding_val_suffix = 'sliding_spaced_fixedgaussian0.1_new_extrapolation_0.2_0.4'

    for p in sliding_path:
        best = np.inf
        best_epoch = 0
        args.suffix = sliding_suffix
        args.val_suffix = sliding_val_suffix
        # try:
        #     for i in np.linspace(10, 490, 49):
        # args.weight_path = p+'/'+str(int(i))+'_decoder.pt'
        args.weight_path = p+'/best_decoder.pt'
        loss, truth, pred, condition, valid_data = load_predict(
            args, args.weight_path, stop_ind=np.inf, record=False)
        # print('loss', args.weight_path, np.mean(loss))
        #         if np.mean(loss) < best:
        #             best = np.mean(loss)
        #             best_epoch = i
        #         # print(i, args.weight_path, args.val_suffix, np.mean(loss))
        #     print('best', best, best_epoch, p)
        # except Exception as e:
        #     print(e)
        # print('best', best, best_epoch, p)

    # loss, truth, pred, condition, valid_data = load_predict(
    #     args, args.weight_path, stop_ind=np.inf)


if __name__ == "__main__":
    test()
