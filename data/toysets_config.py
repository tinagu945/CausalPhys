#[num_sims, num_atoms, num_timesteps, num_dims] 5x5x5x5,5,4,1
step=19
interval=2
data=[]
ass = [1,22,3, -4,500,-1000,90, -87,333,43,32,-1.8,1.2]
bss = [0.1,1.9, 6,10,55,701.3,80,3,90,54,45,-99,0,-1,2.3]
css= [-3,-2,100,50,51,1.6,1.2,1.5,23,54,-17,0,29,90]
dss = [-10,-9,-8,0,1,268,69,199,20,-20,-34,63,70,101]
# ess = [255, 78,65,4,1]
# fss = [0, 0.1,0.2,0.3,50]
for a in ass:
    A=[[a,1,0,0,0,0,0,0]]*step
    for b in bss:
        B=[[b,0,1,0,0,0,0,0]]*step
        for c in css:
            C = [[c,0,0,1,0,0,0,0]]*step
            for d in dss:
                D = [[d,0,0,0,1,0,0,0]]*step
                E,F,G = np.zeros((step, 8)),np.zeros((step, 8)),np.zeros((step, 8))
                E[:,1:]=[0,0,0,0,1,0,0]
                F[:,1:]=[0,0,0,0,0,1,0]
                G[:,1:]=[0,0,0,0,0,0,1]
                for i in range(1, step):
                    E[i, 0]= E[i-1, 0]+a
                    F[i, 0] = F[i-1, 0]+b
                    #1
#                     G[i, 0] =  E[i-1, 0] * F[i-1, 0]
                    #2
                    G[i, 0] =  G[i-1, 0]+E[i-1, 0] * F[i-1, 0]
                    #3
#                     G[i, 0] =  G[i-1, 0]+E[i, 0] * F[i, 0]
                    #4
#                     G[i, 0] =  E[i-1, 0] / (F[i-1, 0]+1)
                    #5
                    G[i, 0] =  G[i-1, 0]+ E[i-1, 0] / (F[i-1, 0]+1)
                    #6
#                     G[i, 0] =  G[i-1, 0]+E[i, 0] / (F[i, 0]+1)
                    #7
                    data_new[:,-3,i,0] =data_new[:,-3,i-1,0]+ data_new[:,0,i-1,0]
                    F = F_t-1 + B -2C
                    G = G_t-1 + 0.5*C
    #                     data_new[:,-2,i,0] = data_new[:,-2,i-1,]+ data_new[:,1,i-1,0]-2*data_new[:,2,i-1,0]
                    
                    data_new[:,-1,i,0] = data_new[:,-1,i-1,0]+ 0.5*data_new[:,2,i-1,0]
                    #8
#                     data_new[:,-1,i,0] = data_new[:,-1,i-1,0]+data_new[:,-3,i-1,0]/(data_new[:,2,i-1,0]+1)

# A=B+C, BCDE -> A?
# plot graph of prediction
# other environment: spring? 


                trial=np.stack([A,B,C,D,E,F,G],axis=0) 
                data.append(trial)

  
#session 0: 2 7, 0 vel w/ loc->vel=0
#session 2: 0 2, 3 vel, 5 5
