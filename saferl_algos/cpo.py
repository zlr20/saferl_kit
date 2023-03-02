import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from saferl_utils import Critic,CPO_Critic, Stochastic_Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

class CPO(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        eval_env,
        rew_discount=0.99,
        cost_discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        expl_noise = 0.1,
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
        
        # evaluation environment
        self.eval_env = eval_env


    def select_action(self, state,exploration=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if exploration:
            noise = np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action
    
    
    def train(self, replay_buffer, batch_size=256, runtime_logger=None):
        # Sample replay buffer 
        state, action, next_state, reward, cost, cost_to_go, reward_to_go, not_done = replay_buffer.sample(batch_size)
        
        # train the cost critic, and reward critics 
        with torch.no_grad():
            next_action = (self.actor(next_state)).clamp(-self.max_action, self.max_action)
            
            # train the cost critic 
            # target_QC
            target_QC = cost + not_done * self.cost_discount * self.cost_critic.Q(next_state, next_action)
            current_QC = self.cost_critic.Q(state, action)
            # train the reward critic
            # target_RC
            target_RC = reward + not_done * self.rew_discount * self.reward_critic.Q(next_state, next_action)
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
                    
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            
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
            jc, _ = runtime_logger.get_stats("EpCost") # mean EpCost over all episodes
            
            # Objective function
            action = self.actor(state)
            reward_objective = - (self.reward_critic.Q(state, action) - self.reward_critic.V(state)).mean() # max A(s,a) 
            
            # Surrogate cost function 
            action = self.actor(state)
            # ! not considering the discound factor for now
            advc = (self.cost_critic.Q(state, action) - self.cost_critic.V(state)).mean() # cost advantage with respect policy
            
            # get the gradient of objective function
            g = torch.autograd.grad(reward_objective, self.actor.parameters())
            
            # get the gradient of surrogate cost function
            b = torch.autograd.grad(advc, self.actor.parameters())
            
            # get the surrogate cost old 
            actor_old = copy.deepcopy(self.actor)
            cost_critic_old = copy.deepcopy(self.cost_critic)
            surr_cost_old = (cost_critic_old.Q(state, actor_old(state)) - cost_critic_old.V(state)).mean() # surrogate cost with respect to current policy
            
            # get the Hessian 
            kl_div = torch.distributions.kl.kl_divergence(self.actor.norm(), actor_old.norm()).sum()
            Jaccobi = torch.autograd.grad(kl_div, self.actor.parameters())
            Hessian = torch.autograd.grad(Jaccobi, self.actor.parameters())
            

            # ! RUIC start
            # todo: update c, rescale
            EpCost = 'todo'
            cost_lim = 'todo'
            EpLen = 'todo'
            c = EpCost - cost_lim
            rescale = EpLen
            
            # c + rescale * b^T (theta - theta_k) <= 0, equiv c/rescale + b^T(...)
            c /= (rescale + EPS)
            
            # Core calculations for CPO
            v = tro.cg(Hx, g) # v = H \ g
            approx_g = Hx(v) # g
            q = np.dot(v, approx_g) # vT @ g

            # todo: solve QP
            # todo: decide optimization cases (feas/infeas, recovery)
            # todo: get optimal theta-theta_k direction
            # todo: line search to find theta under surrogate constraints

            # # Compute actor loss
            # action = self.actor(state)
            # actor_loss = - self.critic.Q1(state, action).mean()
            
            # # Optimize the actor 
            # self.actor_optimizer.zero_grad()
            # actor_loss.backward()
            # self.actor_optimizer.step()

            # # Update the frozen target models
            # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
            pass
            
        
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
 
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, seed, flag, eval_episodes=5):
    avg_reward = 0.
    avg_cost = 0.
    for _ in range(eval_episodes):
        if flag == 'constraint_violation':
            reset_info, done = eval_env.reset(), False
            state = reset_info[0]
        else:
            state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            avg_reward += reward
            
            # if info[flag]!=0:
            #     avg_cost += 1
            if flag == 'safety_gym':
                # avg_cost += info['cost_hazards']
                avg_cost += info['cost']
            else:
                if info[flag]!=0:
                    avg_cost += 1

    avg_reward /= eval_episodes
    avg_cost /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} Cost {avg_cost:.3f}.")
    print("---------------------------------------")
    return avg_reward,avg_cost