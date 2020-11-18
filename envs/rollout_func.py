import torch

#The #input and #target nodes, physics equation, each nodes' value range, etc. should together be an environment.
def rollout_sliding_cube(inputs, num_outputs, g=9.8, delta=True, interval=2):
    mu = inputs[:,2,0,0]
    theta=inputs[:,3,0,0] 
    trajectory_len = inputs.size(2)
    
    # 2 outputs: vel and loc
    outputs = torch.zeros((inputs.size(0), num_outputs, trajectory_len, 1), device=inputs.device)

    for i in range(1,trajectory_len):
        outputs[:,0,i,0] = outputs[:,0,i-1,0]+g*(torch.sin(theta)-torch.cos(theta)*mu)
        if delta:
            outputs[:,1,i,0]= outputs[:,0,i-1,0]*interval+0.5*g*(torch.sin(theta)-torch.cos(theta)*mu)*(interval**2)
        else:
            outputs[:,1,i,0]= outputs[:,1,i-1,0]+ data_new[:,0,i-1,0]*interval+0.5*g*(torch.sin(theta)-torch.cos(theta)*mu)*(interval**2)

    return outputs        
                
                
                
                
