import torch
import numpy as np 

def compute_loss_vc():
    # obs, cost_ret = data['obs'], data['cost_ret']
    obs = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [13, 14, 15] ,[13, 14, 15] ,[13, 14, 15] , [13, 14, 15] ,[13, 14, 15]])
    cost_ret = torch.tensor([0., 1.2, 5., 0, 0, 0, 0, 0, 0, 0])
    
    # down sample the imbalanced data 
    cost_ret_positive = cost_ret[cost_ret > 0]
    obs_positive = obs[cost_ret > 0]
    
    cost_ret_zero = cost_ret[cost_ret == 0]
    obs_zero = obs[cost_ret == 0]
    
     
    frac = len(cost_ret_positive) / len(cost_ret_zero) 
    if frac < 1.:# Fraction of elements to keep
        indices = np.random.choice(len(cost_ret_zero), size=int(len(cost_ret_zero)*frac), replace=False)
        cost_ret_zero_downsample = cost_ret_zero[indices]
        obs_zero_downsample = obs_zero[indices]
        
        # concatenate 
        obs_downsample = torch.cat((obs_positive, obs_zero_downsample), dim=0)
        cost_ret_downsample = torch.cat((cost_ret_positive, cost_ret_zero_downsample), dim=0)
    else:
        # no need to downsample 
        obs_downsample = obs
        cost_ret_downsample = cost_ret
    
    # downsample cost return zero 
    print(f'obs_downsample is {obs_downsample}')
    print(f'cost_ret_downsample is {cost_ret_downsample}')

if __name__ == "__main__":
    compute_loss_vc()