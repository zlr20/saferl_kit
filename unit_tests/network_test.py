import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy 

class Stochastic_Actor(nn.Module):
    def __init__(self):
        super(Stochastic_Actor, self).__init__()

        state_dim = 4
        action_dim = 2
        max_action = 2
        self.mul1 = nn.Linear(state_dim, 256)
        self.mul2 = nn.Linear(256, 256)
        self.mul3 = nn.Linear(256, action_dim)
        
        self.std1 = nn.Linear(state_dim, 256)
        self.std2 = nn.Linear(256, 256)
        self.std3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def get_norm(self, state):
        mu = F.relu(self.mul1(state))
        mu = F.relu(self.mul2(mu))
        mu = F.sigmoid(self.mul3(mu))
        
        std = F.relu(self.std1(state))
        std = F.relu(self.std2(std))
        std = 0.1 + 0.9 * F.sigmoid(self.std3(std))
        
        norm = torch.distributions.Normal(mu, std)
        
        return norm

    def forward(self, state):
        
        norm = self.get_norm(state)
        
        # return self.max_action * norm.rsample((state.shape[0],))
        return self.max_action * norm.rsample()
    
    
    
actor = Stochastic_Actor()
state = torch.FloatTensor(np.random.rand(10,4))
actor_old = copy.deepcopy(actor)

kl_div = torch.distributions.kl.kl_divergence(actor.get_norm(state), actor_old.get_norm(state)).sum(axis=1).mean()
import pdb; pdb.set_trace()