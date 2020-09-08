import time
import torch
from utils.functions import *
from utils.logger import Logger
from val import val_control


def train_val_control(args, log_prior, logger, optimizer, save_folder, train_loader, valid_data_loader,
                      decoder, rel_rec, rel_send, data_trained, mask_grad=False, train_log=5, val_log=1):
    """
        train_log (int, optional): When the amount of data seen divides train_log, do a log. Defaults to 5.
        val_log (int, optional): Similar to above. Defaults to 1.
    """
    if data_trained == 0:
        print('Doing initial validation before training...')
        val_control(
            args, log_prior, logger, save_folder, valid_data_loader, -1, decoder, rel_rec, rel_send)

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
    a_train = []
    b_train = []
    c_train = []
    control_train = []

    rel_graphs = []
    rel_graphs_grad = []
    msg_hook_mean = []

    decoder.train()
    for batch_idx, all_data in enumerate(train_loader):
        if args.grouped:
            # edge is only for calculating edge accuracy. Since we have not included that, edge is not used.
            data, which_node, edge = all_data[0].cuda(
            ), all_data[1].cuda(), all_data[2].cuda()
            output, logits, msg_hook = decoder(data, rel_rec, rel_send,
                                               args.temp, args.hard, args.prediction_steps, [])
            control_constraint_loss = control_loss(
                msg_hook, which_node, args.input_atoms, args.variations)*args.control_constraint
            # if batch_idx == 20:
            #     print('start3', time.time()-start3)

        else:
            data, edge = all_data[0].cuda(), all_data[1].cuda()
            output, logits, msg_hook = decoder(data, rel_rec, rel_send,
                                               args.temp, args.hard, args.prediction_steps, [])
            control_constraint_loss = torch.zeros(1).cuda()

        data_trained += data.size(0)
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
        nll_train_lasttwo.append(loss_nll_lasttwo.item())
        nll_train_lasttwo_5.append(loss_nll_lasttwo_series[5].item())
        nll_train_lasttwo_10.append(loss_nll_lasttwo_series[10].item())
        nll_train_lasttwo__1.append(loss_nll_lasttwo_series[-1].item())
        nll_train_lasttwo_1.append(loss_nll_lasttwo_series[1].item())

        kl_train.append(loss_kl.item())
        a_train.append(F.mse_loss(
            output[:, -1, :, :], target[:, -1, :, :]).item())
        b_train.append(F.mse_loss(
            output[:, -2, :, :], target[:, -2, :, :]).item())
        c_train.append(F.mse_loss(
            output[:, -3, :, :], target[:, -3, :, :]).item())
        control_train.append(control_constraint_loss.item())
        msg_hook_mean.append(msg_hook.mean(dim=1).sum().item())
        if batch_idx % 50 == 0:
            print(batch_idx)
            print('Train', control_constraint_loss.item(),
                  loss_kl.item(), loss_nll.item(), loss_nll_lasttwo.item(), loss_nll_lasttwo_series[0].item())
            if args.gt_A is False:
                rel_graphs.append(decoder.rel_graph.detach().cpu().numpy())
                rel_graphs_grad.append(
                    decoder.rel_graph.grad.detach().cpu().numpy())

        if data_trained % train_log == 0:
            print('Train logging...')
            logger.log('train', decoder, data_trained//train_log, np.mean(nll_train), np.mean(nll_train_lasttwo), kl=np.mean(kl_train), mse=np.mean(mse_train), a=np.mean(a_train), b=np.mean(b_train), c=np.mean(
                c_train), control_constraint_loss=np.mean(control_train), lr=optimizer.param_groups[0]['lr'], rel_graphs=rel_graphs, rel_graphs_grad=rel_graphs_grad, msg_hook_weights=np.mean(msg_hook_mean), nll_train_lasttwo=np.mean(nll_train_lasttwo), nll_train_lasttwo_5=np.mean(nll_train_lasttwo_5), nll_train_lasttwo_10=np.mean(nll_train_lasttwo_10), nll_train_lasttwo__1=np.mean(nll_train_lasttwo__1), nll_train_lasttwo_1=np.mean(nll_train_lasttwo_1))

        if data_trained % val_log == 0:
            val_control(
                args, log_prior, logger, save_folder, valid_data_loader, data_trained//val_log, decoder, rel_rec, rel_send)

    print('Train AVG', np.mean(control_train),
          np.mean(kl_train), np.mean(nll_train), np.mean(nll_train_lasttwo))
    print(data_trained//train_log, decoder.rel_graph.softmax(-1),
          decoder.rel_graph.size())
    return data_trained
