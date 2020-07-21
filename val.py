from utils.logger import Logger
from utils.functions import *

def val_control(args, logger, save_folder, valid_loader, epoch, decoder, rel_rec, rel_send):
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []
    a_val =[]
    b_val=[]
    c_val=[]

    decoder.eval()
    for batch_idx, data in enumerate(valid_loader):
        if args.cuda:
            data = data.cuda()

        # validation output uses teacher forcing
        output, logits = decoder(data, rel_rec, rel_send,
                             args.temp, args.hard, 1)
        prob = my_softmax(logits, -1)

        target = data[:, :, 1:, :] 
        loss_nll = nll_gaussian(output, target, args.var)
        a= nll_gaussian(output[:,-1,:,:], target[:,-1,:,:], args.var)
        b= nll_gaussian(output[:,-2,:,:], target[:,-2,:,:], args.var)
        c= nll_gaussian(output[:,-3,:,:], target[:,-3,:,:], args.var)
            
        if args.prior:
            loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        else:
            loss_kl = kl_categorical_uniform(prob, args.num_atoms,
                                             args.edge_types)

        mse_val.append(F.mse_loss(output, target).item())
        nll_val.append(loss_nll.item())
        kl_val.append(loss_kl.item())
        a_val.append(a.item())
        b_val.append(b.item())
        c_val.append(c.item())
        
        logger.log('val', decoder, epoch, nll_val, kl_val, mse_val, a=a_val, b=b_val, c=c_val)

    return np.mean(nll_val)
