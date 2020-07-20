import os
import torch
import numpy as np


class Logger(object):
    def __init__(self, save_folder):
        self.val_writer = val_writer = SummaryWriter(save_folder)
        self.best_val_loss = np.inf
        self.best_epoch = -1
        
    def log(mode, decoder, global_epoch, nll, kl, mse, a, b, c, lr):
        self.val_writer.add_scalar('nll_'+mode,np.mean(nll), global_epoch) 
        self.val_writer.add_scalar('kl_'+mode, np.mean(kl), global_epoch) 
        self.val_writer.add_scalar('mse_'+mode,np.mean(mse), global_epoch) 
        self.val_writer.add_scalar('-1_val',np.mean(a), global_epoch)
        self.val_writer.add_scalar('-2_val',np.mean(b), global_epoch)
        self.val_writer.add_scalar('-3_val',np.mean(v), global_epoch)    
        
        if mode =='train':
            self.val_writer.add_scalar('lr',scheduler.get_lr()[0], global_epoch) 
            
            torch.save(decoder.state_dict(), os.path.join(save_folder, str(global_epoch)+'_decoder.pt'))
            np.save(os.path.join(save_folder, str(global_epoch)+'_rel_graph.npy'), np.array(rel_graphs))
            np.save(os.path.join(save_folder, str(global_epoch)+'_rel_graph_grad.npy'), np.array(rel_graphs_grad))
        elif mode =='val':
            if np.mean(nll_val) < best_val_loss:
                self.best_val_loss = np.mean(nll_val)
                self.best_epoch = global_epoch
                
                print('Best model so far, saving...', file)
                torch.save([decoder.state_dict(), decoder.rel_graph], 'best_decoder.pt')
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
                      'time: {:.4f}s'.format(time.time() - t))
        
            