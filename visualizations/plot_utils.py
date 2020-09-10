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


def load_predict(args, weight_path, stop_ind=5):
    decoder = MLPDecoder_Causal(args).cuda()
    off_diag = np.ones([args.num_atoms, args.num_atoms])
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec).cuda()
    rel_send = torch.FloatTensor(rel_send).cuda()
    suffix= 'valid_causal_vel_'+args.suffix+'_interpolation'

    if args.grouped:
        assert args.train_bs % args.variations == 0, "Grouping training set requires args.traing-bs integer times of args.variations"

        train_data = load_one_graph_data(
            suffix, size=args.train_size, self_loop=args.self_loop, control=True, control_nodes=args.input_atoms, variations=args.variations)
        train_sampler = RandomPytorchSampler(train_data)
        train_data_loader = DataLoader(
            train_data, batch_size=args.train_bs, shuffle=False, sampler=train_sampler)

    else:
        train_data = load_one_graph_data(
            suffix, size=args.train_size, self_loop=args.self_loop, control=False)
        train_data_loader = DataLoader(
            train_data, batch_size=args.train_bs, shuffle=False)

    decoder.load_state_dict(torch.load(weight_path)[0])
    decoder.eval()
    # if args.gt_A:
    #     print('loading rel_graph from', weight_path)
    #     decoder.rel_graph = torch.load(weight_path)[1].cuda()
    loss = []
    truth = []
    pred = []
    condition = []
    for batch_idx, all_data in enumerate(train_data_loader):
        if batch_idx < stop_ind:
            if args.grouped:
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
            print('batch_size', data.size(0))
            target = data[:, :, 1:, :]
            loss_nll, _ = nll_gaussian(
                output[:, -2:, :, :], target[:, -2:, :, :], 5e-5)
            print('Nll', loss_nll)
            loss.append(loss_nll.item())

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
        else:
            print('Avg nll loss', np.mean(loss))
            return loss, truth, pred, condition


def denormalize(target, output, train_data):
    for i in range(target.size(1)):
        output[:, i, :, 0] = (
            (output[:, i, :, 0]+1)*(train_data.maxs[i]-train_data.mins[i]))/2+train_data.mins[i]
        target[:, i, :, 0] = (
            (target[:, i, :, 0]+1)*(train_data.maxs[i]-train_data.mins[i]))/2+train_data.mins[i]
    return target, output
