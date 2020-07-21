import numpy as np


def rollout_sliding_cube(inputs, num_outputs=2, g=9.8, delta=True):
    mu = inputs[:,2,0,0]
    theta=inputs[:,3,0,0] 
    trajectory_len = inputs.size(2)
    # 2 outputs: vel and loc
    outputs = np.zeros((inputs.size(0), num_outputs, trajectory_len, 1), device=inputs.device())

    for i in range(1,trajectory_len):
        outputs[:,0,i,0] = outputs[:,0,i-1,0]+g*(np.sin(theta)-np.cos(theta)*mu)
        if delta:
            outputs[:,1,i,0]= outputs[:,0,i-1,0]*interval+0.5*g*(np.sin(theta)-np.cos(theta)*mu)*(interval**2)
        else:
            outputs[:,1,i,0]= outputs[:,1,i-1,0]+ data_new[:,0,i-1,0]*interval+0.5*g*(np.sin(theta)-np.cos(theta)*mu)*(interval**2)

    return outputs        
                
                
                
                
