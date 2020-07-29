from utils.logger import Logger
from utils.functions import *


def val_control(args, log_prior, logger, save_folder, valid_loader, epoch, decoder, rel_rec, rel_send):
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []
    a_val = []
    b_val = []
    c_val = []

    decoder.eval()
    for batch_idx, all_data in enumerate(valid_loader):
        data, edge = all_data[0].cuda(), all_data[1].cuda()
        output, logits, _ = decoder(data, rel_rec, rel_send,
                                    args.temp, args.hard, args.prediction_steps, [])

        prob = my_softmax(logits, -1)

        target = data[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, args.var)

        loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        loss_kl *= args.kl

        mse_val.append(F.mse_loss(output, target).item())
        nll_val.append(loss_nll.item())
        kl_val.append(loss_kl.item())
        a_val.append(F.mse_loss(
            output[:, -1, :, :], target[:, -1, :, :]).item())
        b_val.append(F.mse_loss(
            output[:, -2, :, :], target[:, -2, :, :]).item())
        c_val.append(F.mse_loss(
            output[:, -3, :, :], target[:, -3, :, :]).item())

    logger.log('val', decoder, epoch, np.mean(nll_val), kl=np.mean(kl_val), mse=np.mean(mse_val), a=np.mean(
        a_val), b=np.mean(b_val), c=np.mean(c_val))

    return np.mean(nll_val)
