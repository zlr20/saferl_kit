import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action

    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))
    
    
class CPO_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CPO_Critic, self).__init__()
        
        # Q function
        self.ql1 = nn.Linear(state_dim + action_dim, 256)
        self.ql2 = nn.Linear(256, 256)
        self.ql3 = nn.Linear(256, 1)
        
        # value function
        self.vl1 = nn.Linear(state_dim, 256)
        self.vl2 = nn.Linear(256, 256)
        self.vl3 = nn.Linear(256, 1)
        
    def Q(self, state, action):
        sa = torch.cat([state, action], 1)

        q = F.relu(self.ql1(sa))
        q = F.relu(self.ql2(q))
        q = self.ql3(q)
        return q

    def V(self, state):
        v = F.relu(self.vl1(state))
        v = F.relu(self.vl2(v))
        v = self.vl3(v)
        return v
        

class C_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(C_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)


    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# predict state-wise \lamda(s)
class MultiplerNet(nn.Module):
    def __init__(self, state_dim):
        super(MultiplerNet, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        #return F.relu(self.l3(a))
        return F.softplus(self.l3(a)) # lagrangian multipliers can not be negative


class Stochastic_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Stochastic_Actor, self).__init__()

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