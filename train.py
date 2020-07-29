import time
import torch
from utils.functions import *
from utils.logger import Logger


def train_control(args, log_prior, logger, optimizer, save_folder, train_loader, epoch,
                  decoder, rel_rec, rel_send, mask_grad=False, log_epoch=10):
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []
    a_train = []
    b_train = []
    c_train = []
    control_train = []

    rel_graphs = []
    rel_graphs_grad = []

    decoder.train()
    for batch_idx, all_data in enumerate(train_loader):
        if args.grouped:
            # edge is only for calculating edge accuracy. Since we have not included that, edge is not used.
            data, which_node, edge = all_data[0].cuda(
            ), all_data[1].cuda(), all_data[2].cuda()
            output, logits, msg_hook = decoder(data, rel_rec, rel_send,
                                               args.temp, args.hard, args.prediction_steps, [])
            control_constraint_loss = control_loss(
                msg_hook, which_node, args.input_atoms)*args.control_constraint

        else:
            data, edge = all_data[0].cuda(), all_data[1].cuda()
            output, logits, _ = decoder(data, rel_rec, rel_send,
                                        args.temp, args.hard, args.prediction_steps, [])
            control_constraint_loss = torch.zeros(1).cuda()

        prob = my_softmax(logits, -1)
        # data: bs, #node, #timesteps, dim
        target = data[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, args.var)

        loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        loss_kl *= args.kl
        loss = loss_nll + loss_kl + control_constraint_loss
        # print(control_constraint_loss, loss_kl, loss_nll, data.size())
        optimizer.zero_grad()
        loss.backward()
#         if mask_grad:
#             mask = torch.zeros(decoder.rel_graph.size(), requires_grad=False, device="cuda")
#             mask[:,:,which_node:args.num_atoms**2:args.num_atoms,:]=1
#             #target nodes influence always allowed to flow
#             for i in range(args.input_atoms, args.num_atoms):
#                 mask[:,:,i:args.num_atoms**2:args.num_atoms]=1
# #             print(mask)
#             decoder.rel_graph.grad *= mask

        optimizer.step()
        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        a_train.append(F.mse_loss(
            output[:, -1, :, :], target[:, -1, :, :]).item())
        b_train.append(F.mse_loss(
            output[:, -2, :, :], target[:, -2, :, :]).item())
        c_train.append(F.mse_loss(
            output[:, -3, :, :], target[:, -3, :, :]).item())
        control_train.append(control_constraint_loss.item())
        if batch_idx % 50 == 0:
            print(batch_idx)
            rel_graphs.append(decoder.rel_graph.detach().cpu().numpy())
            rel_graphs_grad.append(
                decoder.rel_graph.grad.detach().cpu().numpy())

    print(epoch, decoder.rel_graph.softmax(-1), decoder.rel_graph.size())
    if epoch % log_epoch == 0:
        print('Train logging...')
        logger.log('train', decoder, epoch, np.mean(nll_train), kl=np.mean(kl_train), mse=np.mean(mse_train), a=np.mean(a_train), b=np.mean(b_train), c=np.mean(
            c_train), control_constraint_loss=np.mean(control_train), lr=optimizer.param_groups[0]['lr'], rel_graphs=rel_graphs, rel_graphs_grad=rel_graphs_grad)
