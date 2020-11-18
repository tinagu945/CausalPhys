import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, save_folder):
        print('Recording best epoch based on nll_last2!')
        self.save_folder = save_folder
        self.val_writer = SummaryWriter(save_folder)
        self.best_val_loss = np.inf
        self.best_epoch = -1

    def log(self, mode, decoder, epoch, nll, nll_last2, rel_graphs=None, rel_graphs_grad=None, scheduler=None, **kwargs):
        self.val_writer.add_scalar('nll_'+mode, nll, epoch)
        for key, value in kwargs.items():
            self.val_writer.add_scalar(key+'_'+mode, value, epoch)
        print(self.save_folder)
        if mode == 'train':
            torch.save([decoder.state_dict(), decoder.rel_graph], os.path.join(
                self.save_folder, str(epoch)+'_decoder.pt'))
#             np.save(os.path.join(self.save_folder, str(epoch)+'_rel_graph.npy'), np.array(decoder.rel_graph))
            if rel_graphs_grad:
                np.save(os.path.join(self.save_folder, str(epoch) +
                                     '_rel_graph_grad.npy'), np.array(rel_graphs_grad))
        elif mode == 'val':
            self.val_writer.add_scalar(
                'best_epoch_val', self.best_epoch, epoch)
            self.val_writer.add_scalar(
                'best_epoch_loss_val', self.best_val_loss, epoch)
            if nll_last2 < self.best_val_loss:
                self.best_val_loss = np.mean(nll_last2)
                self.best_epoch = epoch
                print('Best model so far, saving...',
                      self.best_val_loss, self.best_epoch)
                torch.save([decoder.state_dict(), decoder.rel_graph, scheduler.state_dict()],
                           os.path.join(
                    self.save_folder, 'best_decoder.pt'))
