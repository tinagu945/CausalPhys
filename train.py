import time
import torch
from utils.functions import *
from utils.logger import Logger 

def train_control(args, logger, optimizer, save_folder,train_loader, epoch, \
                  decoder, rel_rec, rel_send, mask_grad=False):
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []
    a_train =[]
    b_train =[]
    c_train =[]
    
    rel_graphs=[]
    rel_graphs_grad=[]
    
    decoder.train()
    for batch_idx, (data, which_node) in enumerate(train_loader):
        if args.cuda:
            data, which_node = data.cuda(), which_node.cuda()

        output, logits = decoder(data, rel_rec, rel_send,
                         args.temp, args.hard, args.prediction_steps)
        prob = my_softmax(logits, -1)

        #data: bs, #node, #timesteps, dim
        target = data[:, :, 1:, :] 
        loss_nll = nll_gaussian(output, target, args.var)

        if args.prior:        
            loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        else:
            loss_kl = kl_categorical_uniform(prob, args.num_atoms,
                                             args.edge_types)
        loss_kl *= args.kl
        loss = loss_nll +loss_kl

        a= nll_gaussian(output[:,-1,:,:], target[:,-1,:,:], args.var)
        b= nll_gaussian(output[:,-2,:,:], target[:,-2,:,:], args.var)
        c= nll_gaussian(output[:,-3,:,:], target[:,-3,:,:], args.var)

        optimizer.zero_grad()
        loss.backward()
        #TODO: hard coded for now!
        if mask_grad:
            mask = torch.zeros(decoder.rel_graph.size(), requires_grad=False, device="cuda")
            mask[:,:,(5-which_node):args.num_atoms**2:args.num_atoms]=1
            mask[:,:,6:args.num_atoms**2:args.num_atoms]=1
            mask[:,:,7:args.num_atoms**2:args.num_atoms]=1               
            decoder.rel_graph.grad *= mask
  
        optimizer.step()    
        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        a_train.append(a.item())
        b_train.append(b.item())
        c_train.append(c.item())
        
        rel_graphs.append(decoder.rel_graph.detach().cpu().numpy())
        rel_graphs_grad.append(decoder.rel_graph.grad.detach().cpu().numpy())
   
    print(epoch, decoder.rel_graph.softmax(-1), decoder.rel_graph.size()) 
    if epoch % 10==0:
        logger.log('train', decoder, epoch, nll_train, kl_train, mse_train,\
                   a=a_train, b=b_train, c=c_train, lr=optimizer.param_groups[0]['lr'])
        