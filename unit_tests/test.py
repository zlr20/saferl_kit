import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.l1 = nn.Linear(2, 3)
        self.l2 = nn.Linear(3, 1)
        
    def forward(self, input):
        a = F.relu(self.l1(input))
        a = F.sigmoid(self.l2(a))
        return a

def get_net_param_np_vec(net):
    """
        Get the parameters of the network as numpy vector
    """
    return torch.cat([val.flatten() for val in net.parameters()], axis=0).detach().cpu().numpy()

def auto_grad_vec(objective, net, to_numpy=True):
    """
        Get the gradient of the objective with respect to the parameters of the network
    """
    grad = torch.autograd.grad(objective, net.parameters(), create_graph=True)
    if to_numpy:
        return torch.cat([val.flatten() for val in grad], axis=0).detach().cpu().numpy()
    else:
        return torch.cat([val.flatten() for val in grad], axis=0)

def auto_hessian_np_vec(objective, net):
    jacob = auto_grad_vec(objective, net, to_numpy=False)
    H = torch.stack([auto_grad_vec(val, net, to_numpy=False) for val in jacob], axis=0).detach().cpu().numpy()
    return H

def assign_net_param_from_flat(param_vec, net):
    param_sizes = [np.prod(list(val.shape)) for val in net.parameters()]
    ptr = 0
    for s, param in zip(param_sizes, net.parameters()):
        param.data.copy_(torch.from_numpy(param_vec[ptr:ptr+s]).reshape(param.shape))
        ptr += s

net = Net()
input = torch.randn(2, 2)
output = net(input).mean()

# jacob = auto_grad_vec(output, net)

v = get_net_param_np_vec(net)
v /= v
print(auto_hessian_np_vec(output, net))

assign_net_param_from_flat(v, net)

print(get_net_param_np_vec(net))
    
import ipdb; ipdb.set_trace()