import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, save_folder):
        self.save_folder = save_folder
        self.val_writer = SummaryWriter(save_folder)
        self.best_val_loss = np.inf
        self.best_epoch = -1
        
    def log(self, mode, decoder, epoch, nll, kl, mse, a=None, b=None, c=None, lr=None):
        self.val_writer.add_scalar('nll_'+mode,np.mean(nll), epoch) 
        self.val_writer.add_scalar('kl_'+mode, np.mean(kl), epoch) 
        self.val_writer.add_scalar('mse_'+mode,np.mean(mse), epoch) 
        if a:
            self.val_writer.add_scalar('-1_val',np.mean(a), epoch)
        if b:
            self.val_writer.add_scalar('-2_val',np.mean(b), epoch)
        if c:
            self.val_writer.add_scalar('-3_val',np.mean(c), epoch)    
        
        if mode =='train':
            self.val_writer.add_scalar('lr',lr, epoch) 
            
            torch.save(decoder.state_dict(), os.path.join(self.save_folder, str(epoch)+'_decoder.pt'))
#             np.save(os.path.join(self.save_folder, str(epoch)+'_rel_graph.npy'), np.array(decoder.rel_graphs))
#             np.save(os.path.join(self.save_folder, str(epoch)+'_rel_graph_grad.npy'), np.array(rel_graphs_grad))
        elif mode =='val':
            if np.mean(nll) < self.best_val_loss:
                self.best_val_loss = np.mean(nll)
                self.best_epoch = epoch            
                print('Best model so far, saving...', self.best_val_loss, self.best_epoch)
                torch.save([decoder.state_dict(), decoder.rel_graph], 'best_decoder.pt')
                
        
            