import time
import torch
from utils.functions import *


def train_control(args, log_prior, optimizer, save_folder, train_loader, valid_data_loader,
                  decoder, epoch, mask_grad=False):

    t = time.time()
    nll_train = []
    nll_train_lasttwo = []
    nll_train_lasttwo_5 = []
    nll_train_lasttwo_10 = []
    nll_train_lasttwo__1 = []
    nll_train_lasttwo_1 = []

    acc_train = []
    kl_train = []
    mse_train = []
    control_train = []

    rel_graphs = []
    rel_graphs_grad = []
    msg_hook_mean = []

    decoder.train()
    for batch_idx, all_data in enumerate(train_loader):
        # if args.grouped:
        #     # edge is only for calculating edge accuracy. Since we have not included that, edge is not used.
        #     data, which_node, edge = all_data[0].cuda(
        #     ), all_data[1].cuda(), all_data[2].cuda()
        #     output, logits, msg_hook = decoder(data)

        #     if args.control_constraint != 0:
        #         control_constraint_loss = control_loss(
        #             msg_hook, which_node, args.input_atoms, args.variations)*args.control_constraint
        #     else:
        #         control_constraint_loss = torch.zeros(1).cuda()
        #     # if batch_idx == 20:
        #     #     print('start3', time.time()-start3)

        # else:
        data, edge = all_data[0].cuda(), all_data[1].cuda()
        # output, logits, msg_hook = decoder(data)
        output, logits, _ = decoder(data)
        control_constraint_loss = torch.zeros(1).cuda()

        prob = my_softmax(logits, -1)
        # data: bs, #node, #timesteps, dim
        target = data[:, :, 1:, :]
        loss_nll_lasttwo, loss_nll_lasttwo_series = nll_gaussian(
            output[:, -2:, :, :], target[:, -2:, :, :], args.var)
        loss_nll, _ = nll_gaussian(output, target, args.var)

        loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        loss_kl *= args.kl
        loss = loss_nll + loss_kl + control_constraint_loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
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
        nll_train_lasttwo.append(loss_nll_lasttwo.item())
        nll_train_lasttwo_5.append(loss_nll_lasttwo_series[5].item())
        nll_train_lasttwo_10.append(loss_nll_lasttwo_series[10].item())
        nll_train_lasttwo__1.append(loss_nll_lasttwo_series[-1].item())
        nll_train_lasttwo_1.append(loss_nll_lasttwo_series[1].item())

        kl_train.append(loss_kl.item())
        control_train.append(control_constraint_loss.item())
        # msg_hook_mean.append(msg_hook.mean(dim=1).sum().item())
        if batch_idx % 50 == 0:
            print('batch_idx', batch_idx)
            print('Train', control_constraint_loss.item(),
                  loss_kl.item(), loss_nll.item(), loss_nll_lasttwo.item(), loss_nll_lasttwo_series[0].item())
            if (not args.gt_A) and (not args.all_connect):
                rel_graphs.append(decoder.rel_graph.detach().cpu().numpy())
                # rel_graphs_grad.append(
                #     decoder.rel_graph.grad.detach().cpu().numpy())

        del loss, output, logits, target, data, prob

    print('Train AVG this epoch', np.mean(control_train),
          np.mean(kl_train), np.mean(nll_train), np.mean(nll_train_lasttwo))
    print('epoch', epoch, decoder.rel_graph.softmax(-1)[:, :, -16:, :],
          decoder.rel_graph.size())

    return np.mean(nll_train), np.mean(nll_train_lasttwo), np.mean(kl_train), np.mean(mse_train), np.mean(control_train), optimizer.param_groups[0]['lr'], rel_graphs, rel_graphs_grad, 0, np.mean(nll_train_lasttwo), np.mean(nll_train_lasttwo_5), np.mean(nll_train_lasttwo_10), np.mean(nll_train_lasttwo__1), np.mean(nll_train_lasttwo_1)
# np.mean(msg_hook_mean)
