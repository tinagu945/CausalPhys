


def val(valid_loader):
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
    args.val_writer.add_scalar('nll_train',np.mean(nll_train), global_epoch) 
    args.val_writer.add_scalar('kl_train',np.mean(kl_train), global_epoch) 
    args.val_writer.add_scalar('mse_train',np.mean(mse_train), global_epoch) 
    args.val_writer.add_scalar('nll_val',np.mean(nll_val), global_epoch) 
    args.val_writer.add_scalar('-1_val',np.mean(a_val), global_epoch)
    args.val_writer.add_scalar('-2_val',np.mean(b_val), global_epoch)
    args.val_writer.add_scalar('-3_val',np.mean(c_val), global_epoch)
    args.val_writer.add_scalar('kl_val',np.mean(kl_val), global_epoch) 
    args.val_writer.add_scalar('mse_val',np.mean(mse_val), global_epoch) 
    args.val_writer.add_scalar('lr',scheduler.get_lr()[0], global_epoch) 
    
    
    
    
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save([decoder.state_dict(), decoder.rel_graph], decoder_file)
        print('Best model so far, saving...', file=log)
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
              'a_val: {:.10f}'.format(np.mean(a_val)),
              'b_val: {:.10f}'.format(np.mean(b_val)),
              'c_val: {:.10f}'.format(np.mean(c_val)),
              'kl_val: {:.10f}'.format(np.mean(kl_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(nll_val)
