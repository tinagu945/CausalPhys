from utils.logger import Logger
from utils.functions import *


def val_control(args, log_prior, logger, save_folder, valid_loader, epoch, decoder, rel_rec, rel_send):
    nll_val = []
    nll_val_lasttwo = []
    nll_val_lasttwo_1 = []
    nll_val_lasttwo_5 = []
    nll_val_lasttwo_10 = []
    nll_val_lasttwo__1 = []

    acc_val = []
    kl_val = []
    mse_val = []
    a_val = []
    b_val = []
    c_val = []
    control_val = []
    msg_hook_mean = []

    decoder.eval()
    for batch_idx, all_data in enumerate(valid_loader):
        if args.val_grouped:
            data, which_node, edge = all_data[0].cuda(
            ), all_data[1].cuda(), all_data[2].cuda()
            # import pdb
            # pdb.set_trace()
            output, logits, msg_hook = decoder(data, rel_rec, rel_send,
                                               args.temp, args.hard, args.prediction_steps, [])

            control_constraint_loss = control_loss(
                msg_hook, which_node, args.input_atoms, args.val_variations)
        else:
            data, edge = all_data[0].cuda(), all_data[1].cuda()
            output, logits, msg_hook = decoder(data, rel_rec, rel_send,
                                               args.temp, args.hard, args.prediction_steps, [])
            control_constraint_loss = torch.zeros(1).cuda()

        # print('val', data[:, :-2, 0, 0])
        prob = my_softmax(logits, -1)

        target = data[:, :, 1:, :]
        loss_nll, _ = nll_gaussian(output, target, args.var)
        loss_nll_lasttwo, loss_nll_lasttwo_series = nll_gaussian(
            output[:, -2:, :, :], target[:, -2:, :, :], args.var)

        loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        loss_kl *= args.kl

        if batch_idx < 5:
            print('Val', control_constraint_loss.item(),
                  loss_kl.item(), loss_nll.item(), loss_nll_lasttwo.item(), loss_nll_lasttwo_series[0].item())

        mse_val.append(F.mse_loss(output, target).item())
        nll_val.append(loss_nll.item())
        nll_val_lasttwo.append(loss_nll_lasttwo.item())
        nll_val_lasttwo_5.append(loss_nll_lasttwo_series[5].item())
        nll_val_lasttwo_10.append(loss_nll_lasttwo_series[10].item())
        nll_val_lasttwo__1.append(loss_nll_lasttwo_series[-1].item())
        nll_val_lasttwo_1.append(loss_nll_lasttwo_series[1].item())

        kl_val.append(loss_kl.item())
        a_val.append(F.mse_loss(
            output[:, -1, :, :], target[:, -1, :, :]).item())
        b_val.append(F.mse_loss(
            output[:, -2, :, :], target[:, -2, :, :]).item())
        c_val.append(F.mse_loss(
            output[:, -3, :, :], target[:, -3, :, :]).item())
        control_val.append(control_constraint_loss.item())
        msg_hook_mean.append(msg_hook.mean(dim=1).sum().item())

    print('Val AVG', np.mean(control_val),
          np.mean(kl_val), np.mean(nll_val), np.mean(nll_val_lasttwo))
    logger.log('val', decoder, epoch, np.mean(nll_val), np.mean(nll_val_lasttwo), kl=np.mean(kl_val), mse=np.mean(mse_val), a=np.mean(
        a_val), b=np.mean(b_val), c=np.mean(c_val), control_constraint_loss=np.mean(control_val), msg_hook_weights=np.mean(msg_hook_mean), nll_val_lasttwo=np.mean(nll_val_lasttwo), nll_val_lasttwo_5=np.mean(nll_val_lasttwo_5), nll_val_lasttwo_10=np.mean(nll_val_lasttwo_10), nll_val_lasttwo__1=np.mean(nll_val_lasttwo__1), nll_val_lasttwo_1=np.mean(nll_val_lasttwo_1))

    return np.mean(nll_val)
