import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from saferl_utils import Critic,CPO_Critic, Stochastic_Actor
from saferl_plotter import log_utils as lu

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:1')
EPS = 1e-8

"""
Conjugate gradient
"""

def get_net_param_np_vec(net):
    """
        Get the parameters of the network as numpy vector
    """
    return torch.cat([val.flatten() for val in net.parameters()], axis=0).detach().cpu().numpy()

def auto_grad(objective, net, to_numpy=True):
    """
        Get the gradient of the objective with respect to the parameters of the network
    """
    grad = torch.autograd.grad(objective, net.parameters(), create_graph=True)
    if to_numpy:
        return torch.cat([val.flatten() for val in grad], axis=0).detach().cpu().numpy()
    else:
        return torch.cat([val.flatten() for val in grad], axis=0)
    
    
def auto_hession_x(objective, net, x):
    jacob = auto_grad(objective, net, to_numpy=False)
    
    return auto_grad(torch.dot(jacob, x), net, to_numpy=True)

def auto_hessian(objective, net):
    jacob = auto_grad(objective, net, to_numpy=False)
    H = torch.stack([auto_grad(val, net, to_numpy=False) for val in jacob], axis=0).detach().cpu().numpy()
    return H

def assign_net_param_from_flat(param_vec, net):
    param_sizes = [np.prod(list(val.shape)) for val in net.parameters()]
    ptr = 0
    for s, param in zip(param_sizes, net.parameters()):
        param.data.copy_(torch.from_numpy(param_vec[ptr:ptr+s]).reshape(param.shape))
        ptr += s

def cg(Ax, b, cg_iters=10):
    x = np.zeros_like(b)
    r = b.copy() # Note: should be 'b - Ax', but for x=0, Ax=0. Change if doing warm start.
    p = r.copy()
    r_dot_old = np.dot(r,r)
    for _ in range(cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + EPS)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r,r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x

class CPO(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        rew_discount=0.99,
        cost_discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        expl_noise = 0.1,
        cost_lim = 1, # ! tune
        delta = 0.1, # ! cannot be too small, otherwise always infeasible recovery
        backtrack_coeff = 0.8,
        backtrack_iters = 10
    ):

        # self.actor = Actor(state_dim, action_dim, max_action).to(device)
        # self.actor_target = copy.deepcopy(self.actor)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.actor = Stochastic_Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # self.critic = Critic(state_dim, action_dim).to(device)
        # self.critic_target = copy.deepcopy(self.critic)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.cost_critic = CPO_Critic(state_dim, action_dim).to(device)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=3e-4)
        
        self.reward_critic = CPO_Critic(state_dim, action_dim).to(device)
        self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=3e-4)

        self.action_dim = action_dim
        self.max_action = max_action
        self.rew_discount = rew_discount
        self.cost_discount = cost_discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.expl_noise = expl_noise

        self.total_it = 0
        
        self.cost_lim = cost_lim
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_iters = backtrack_iters
        self.delta = delta

    def select_action(self, state,exploration=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        # if exploration:
        #     noise = np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
        #     action = (action + noise).clip(-self.max_action, self.max_action)
        return action
    
    
    def train_vanilla_pg_critic(self, replay_buffer, batch_size=256, runtime_logger=None):
        """
        Train Vanilla Policy Gradient Q function and value function
        """
        # Sample replay buffer 
        state, action, next_state, reward, cost, cost_to_go, reward_to_go, not_done = replay_buffer.sample(batch_size)
        
        # train the cost critic, and reward critics 
        with torch.no_grad():
            next_action = (self.actor(next_state)).clamp(-self.max_action, self.max_action)
            
            # train the reward critic
            # target_RC
            target_RC = reward + not_done * self.rew_discount * self.reward_critic.Q(next_state, next_action)
        
        current_RC = self.reward_critic.Q(state, action)
        
        # compute the reawrd critic loss
        critic_QR_loss = F.mse_loss(current_RC, target_RC)
        # compute the reward value function loss 
        value_VR_loss = F.mse_loss(self.reward_critic.V(state), reward_to_go)
        
        # optimize the reward critic
        critic_R_loss = critic_QR_loss + value_VR_loss
        self.reward_critic_optimizer.zero_grad()
        critic_R_loss.backward()
        self.reward_critic_optimizer.step()
    
    
    def train_vanilla_pg(self, replay_buffer, batch_size=256):
        """
        testing training using Vanilla Policy Gradient to only train the policy to maximize the reward
        """
        # Sample replay buffer 
        state, action, next_state, reward, cost, cost_to_go, reward_to_go, not_done = replay_buffer.sample(batch_size)
        
        # optimize the actor such that it maximizes the advantage of the reward critic
        action_tensor = self.actor(state)
        A_pi_k = (self.reward_critic.Q(state, action_tensor) - self.reward_critic.V(state)).mean()
        reward_objective = - A_pi_k  # max A(s,a) 
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        reward_objective.backward()
        self.actor_optimizer.step()
    
    def train_cpo_critic(self, replay_buffer, batch_size=256):
        # Sample replay buffer 
        state, action, next_state, reward, cost, cost_to_go, reward_to_go, not_done = replay_buffer.sample(batch_size)
        
        # train the cost critic, and reward critics 
        with torch.no_grad():
            next_action = (self.actor(next_state)).clamp(-self.max_action, self.max_action)
            
            # train the cost critic 
            # target_QC
            target_QC = cost + not_done * self.cost_discount * self.cost_critic.Q(next_state, next_action)
            # train the reward critic
            # target_RC
            target_RC = reward + not_done * self.rew_discount * self.reward_critic.Q(next_state, next_action)
        
        current_QC = self.cost_critic.Q(state, action)
        current_RC = self.reward_critic.Q(state, action)
            
        # compute the cost critic loss
        # todo: define the new replay buffer that also stores the cost-to-go, reward-to-go
        # todo: implement the value function fitting 
        critic_QC_loss = F.mse_loss(current_QC, target_QC)
        # compute the cost value function loss 
        value_VC_loss = F.mse_loss(self.cost_critic.V(state), cost_to_go)
        
        # compute the reawrd critic loss
        critic_QR_loss = F.mse_loss(current_RC, target_RC)
        # compute the reward value function loss 
        value_VR_loss = F.mse_loss(self.reward_critic.V(state), reward_to_go)
            
        # Optimize the cost critic 
        critic_C_loss = critic_QC_loss + value_VC_loss
        self.cost_critic_optimizer.zero_grad()
        critic_C_loss.backward()
        self.cost_critic_optimizer.step()
        
        # optimize the reward critic
        critic_R_loss = critic_QR_loss + value_VR_loss
        self.reward_critic_optimizer.zero_grad()
        critic_R_loss.backward()
        self.reward_critic_optimizer.step()
        
    
    def train_cpo_policy(self, replay_buffer, batch_size=256, runtime_logger=None):
        # Sample replay buffer 
        state, action, next_state, reward, cost, cost_to_go, reward_to_go, not_done = replay_buffer.sample(batch_size)
        
        # train the cost critic, and reward critics 
        # with torch.no_grad():
        #     next_action = (self.actor(next_state)).clamp(-self.max_action, self.max_action)
            
        #     # train the cost critic 
        #     # target_QC
        #     target_QC = cost + not_done * self.cost_discount * self.cost_critic.Q(next_state, next_action)
        #     # train the reward critic
        #     # target_RC
        #     target_RC = reward + not_done * self.rew_discount * self.reward_critic.Q(next_state, next_action)
        
        # current_QC = self.cost_critic.Q(state, action)
        # current_RC = self.reward_critic.Q(state, action)
            
        # # compute the cost critic loss
        # # todo: define the new replay buffer that also stores the cost-to-go, reward-to-go
        # # todo: implement the value function fitting 
        # critic_QC_loss = F.mse_loss(current_QC, target_QC)
        # # compute the cost value function loss 
        # value_VC_loss = F.mse_loss(self.cost_critic.V(state), cost_to_go)
        
        # # compute the reawrd critic loss
        # critic_QR_loss = F.mse_loss(current_RC, target_RC)
        # # compute the reward value function loss 
        # value_VR_loss = F.mse_loss(self.reward_critic.V(state), reward_to_go)
            
        # # Optimize the cost critic 
        # critic_C_loss = critic_QC_loss + value_VC_loss
        # self.cost_critic_optimizer.zero_grad()
        # critic_C_loss.backward()
        # self.cost_critic_optimizer.step()
        
        # # optimize the reward critic
        # critic_R_loss = critic_QR_loss + value_VR_loss
        # self.reward_critic_optimizer.zero_grad()
        # critic_R_loss.backward()
        # self.reward_critic_optimizer.step()
                    
        # Delayed policy updates
        # if self.total_it % self.policy_freq == 0:
        if True:
            
            # todo: extend buff to include history adv/cadv
            # todo: make sure all tensors are stored
            # todo: logger add episode cost after each episode
            # todo: logger take average of all the stored episode costs so far
            # todo: get (tensor) pi_loss, surr_cost (A_C), d_kl
            # todo: get (value) H, g, b
            # todo: get (value) pi_loss_old, surr_cost_old
            # ! J_C(pi_k) = mean(cost_episodes)
            # ! current repo: J_C(pi_k) is correct from last episode
            # !               A_C(pi_k) uses all past k iterations
            
            # construct the objective function and surrogate cost function
            J_pi_k_C, _ = runtime_logger.get_stats("EpCost") # mean EpCost over all episodes
            
            # Objective function
            action = self.actor(state)
            A_pi_k = (self.reward_critic.Q(state, action) - self.reward_critic.V(state)).mean()
            reward_objective = - A_pi_k  # max A(s,a) 
            
            # Surrogate cost function 
            action = self.actor(state)
            # ! not considering the discound factor for now
            A_pi_k_C = (self.cost_critic.Q(state, action) - self.cost_critic.V(state)).mean() # cost advantage with respect policy
            
            # get the gradient of objective function
            g = auto_grad(reward_objective, self.actor)
            
            # get the gradient of surrogate cost function
            b = auto_grad(A_pi_k_C, self.actor)
            
            # get the surrogate cost old 
            # cost_critic_old = copy.deepcopy(self.cost_critic)
            A_pi_k_C_old = (self.cost_critic.Q(state, self.actor(state)) - self.cost_critic.V(state)).mean() # surrogate cost with respect to current policy
            
            # get the Hessian 
            actor_old = copy.deepcopy(self.actor)
            kl_div = torch.distributions.kl.kl_divergence(self.actor.get_norm(state), actor_old.get_norm(state)).sum(axis=1).mean()
            # define the function handle for the Hessian @ x 
            Hx = lambda x: auto_hession_x(kl_div, self.actor, torch.FloatTensor(x).to(device))
            
            EpCost   = J_pi_k_C
            EpLen, _ = runtime_logger.get_stats("EpLen")
            pi_l_old = reward_objective
            surr_cost_old = A_pi_k_C_old
            
            c        = EpCost - self.cost_lim
            rescale  = EpLen # ? seems to act like std of cost adv
            
            # c + rescale * b^T (theta - theta_k) <= 0, equiv c/rescale + b^T(...)
            
            c /= (rescale + EPS)
            
            # Core calculations for CPO
            Hinv_g   = cg(Hx, g)             # Hinv_g = H \ g        
            approx_g = Hx(Hinv_g)           # g
            q        = Hinv_g.T @ approx_g  # g.T / H @ g

            # solve QP
            # decide optimization cases (feas/infeas, recovery)
            # Determine optim_case (switch condition for calculation,
            # based on geometry of constrained optimization problem)
            if b.T @ b <= 1e-8 and c < 0:
                Hinv_b, r, s, A, B = 0, 0, 0, 0, 0
                optim_case = 4
            else:
                # cost grad is nonzero: CPO update!
                Hinv_b = cg(Hx, b)                # H^{-1} b
                r = Hinv_b.T @ approx_g          # b^T H^{-1} g
                s = Hinv_b.T @ Hx(Hinv_b)        # b^T H^{-1} b
                A = q - r**2 / s            # should be always positive (Cauchy-Shwarz)
                # delta = self.delta          # the the constraint for KL divergence 
                B = 2*self.delta - c**2 / s  # does safety boundary intersect trust region? (positive = yes)

                # c < 0: feasible

                if c < 0 and B < 0:
                    # point in trust region is feasible and safety boundary doesn't intersect
                    # ==> entire trust region is feasible
                    optim_case = 3
                elif c < 0 and B >= 0:
                    # x = 0 is feasible and safety boundary intersects
                    # ==> most of trust region is feasible
                    optim_case = 2
                elif c >= 0 and B >= 0:
                    # x = 0 is infeasible and safety boundary intersects
                    # ==> part of trust region is feasible, recovery possible
                    optim_case = 1
                    print(lu.colorize(f'Alert! Attempting feasible recovery!', 'yellow', bold=True))
                else:
                    # x = 0 infeasible, and safety halfspace is outside trust region
                    # ==> whole trust region is infeasible, try to fail gracefully
                    optim_case = 0
                    print(lu.colorize(f'Alert! Attempting INFEASIBLE recovery!', 'red', bold=True))
            
            print(lu.colorize(f'optim_case: {optim_case}', 'magenta', bold=True))
            
            # get optimal theta-theta_k direction
            if optim_case in [3,4]:
                lam = np.sqrt(q / (2*self.delta))
                nu = 0
            elif optim_case in [1,2]:
                LA, LB = [0, r /c], [r/c, np.inf]
                LA, LB = (LA, LB) if c < 0 else (LB, LA)
                proj = lambda x, L : max(L[0], min(L[1], x))
                lam_a = proj(np.sqrt(A/B), LA)
                lam_b = proj(np.sqrt(q/(2*self.delta)), LB)
                f_a = lambda lam : -0.5 * (A / (lam+EPS) + B * lam) - r*c/(s+EPS)
                f_b = lambda lam : -0.5 * (q / (lam+EPS) + 2 * self.delta * lam)
                lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
                nu = max(0, lam * c - r) / (abs(s+EPS))
            else:
                lam = 0
                nu = np.sqrt(2 * self.delta / (abs(s+EPS)))
            
            # normal step if optim_case > 0, but for optim_case =0,
            # perform infeasible recovery: step to purely decrease cost
            x = (1./(lam+EPS)) * (Hinv_g + nu * Hinv_b) if optim_case > 0 else nu * Hinv_b
            
            # line search to find theta under surrogate constraints
            # CPO uses backtracking linesearch to enforce constraints
            actor_tmp = copy.deepcopy(self.actor)
            
            def set_and_eval(step):
                new_param = get_net_param_np_vec(self.actor) - step * x
                assign_net_param_from_flat(new_param, actor_tmp)
                
                # kl = torch.distributions.kl.kl_divergence(actor_tmp.get_norm(), actor_old.get_norm()).sum()
                kl = torch.distributions.kl.kl_divergence(actor_tmp.get_norm(state), actor_old.get_norm(state)).sum(axis=1).mean()
                action = actor_tmp(state)
                # ! not considering the discound factor for now
                pi_l        = -(self.reward_critic.Q(state, action) - self.reward_critic.V(state)).mean()
                surr_cost   =  (self.cost_critic.Q(state, action) - self.cost_critic.V(state)).mean()
                
                return kl, pi_l, surr_cost
            
            for j in range(self.backtrack_iters):
                
                kl, pi_l_new, surr_cost_new = set_and_eval(self.backtrack_coeff**j)
                
                # set_and_eval(step=self.backtrack_coeff**j)
                print('%d \tkl %.3f \tsurr_cost_new %.3f \tsurr_cost_old %.3f \tpi_l_new %.3f \tpi_l_old %.3f'%(j, kl, surr_cost_new, surr_cost_old, pi_l_new, pi_l_old))
                if (kl <= self.delta and
                    (pi_l_new <= pi_l_old if optim_case > 1 else True) and # if current policy is feasible (optim>1), must preserve pi loss
                    surr_cost_new - surr_cost_old <= max(-c,0)):
                    print(lu.colorize(f'Accepting new params at step %d of line search.'%j, 'green', bold=False))
                    # self.logger.store(BacktrackIters=j)
                    
                    # set the accepted parameters for the actor 
                    new_param = get_net_param_np_vec(self.actor) - self.backtrack_coeff**j * x
                    assign_net_param_from_flat(new_param, self.actor)
                    break

                if j==self.backtrack_iters-1:
                    print(lu.colorize(f'Line search failed! Keeping old params.', 'yellow', bold=False))
                    # self.logger.store(BacktrackIters=j)
                    
                    kl, pi_l_new, surr_cost_new = set_and_eval(0)
            
            print(lu.colorize(f'c (neg->feasible): {c}, EpCost (should <cost_lim): {EpCost}, cost_lim: {self.cost_lim}, B (pos->intersect): {B}, 2delta: {2*self.delta}, c2/s: {c**2 / s}', 'gray', bold=False))


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, seed, flag, eval_episodes=5):
    avg_reward = 0.
    avg_cost = 0.
    for _ in range(eval_episodes):

        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            avg_reward += reward
            avg_cost += info['cost']

    avg_reward /= eval_episodes
    avg_cost /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} Cost {avg_cost:.3f}.")
    print("---------------------------------------")
    return avg_reward,avg_cost