import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from saferl_utils import Critic,Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Dalal 2018 : c_{t} = c_{t-1} + g^T*a_{t}
class C_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(C_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def pred_g(self,state):
        g = F.relu(self.l1(state))
        g = F.relu(self.l2(g))
        return self.l3(g)
    
    def forward(self, state, action):
        g = self.pred_g(state)
        # (B,1,A)x(B,A,1) -> (B,1,1) -> (B,1)
        return torch.bmm(g.unsqueeze(1),action.unsqueeze(2)).view(-1,1)

class TD3Qpsl(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        delta,
        rew_discount=0.99,
        cost_discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        expl_noise = 0.1,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.action_dim = action_dim
        self.max_action = max_action
        self.rew_discount = rew_discount
        self.cost_discount = cost_discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.expl_noise = 0.1

        self.total_it = 0

        self.C_critic = C_Critic(state_dim, action_dim).to(device)
        self.C_critic_target = copy.deepcopy(self.C_critic)
        self.C_critic_optimizer = torch.optim.Adam(self.C_critic.parameters(), lr=3e-4)

        self.delta = delta

    def select_action(self, state, prev_cost=0, use_qpsl=False, exploration=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if use_qpsl:
            action = self.safety_correction(state,action,prev_cost)
        if exploration:
            noise = np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def safety_correction(self,state,action,cost,verbose=False):
        if not torch.is_tensor(state):
            state = torch.tensor(state.reshape(1, -1),requires_grad=False).float().to(device)
        if not torch.is_tensor(action):
            action = torch.tensor(action.reshape(1, -1),requires_grad=False).float().to(device)

        pred = self.C_critic_target(state,action).item() + cost
        if pred <= self.delta:
            return action.cpu().data.numpy().flatten()
        else:
            g = self.C_critic_target.pred_g(state)
            # Equation (5) from Dalal 2018.
            numer = self.C_critic_target(state,action).item() + cost - self.delta
            denomin = torch.bmm(g.unsqueeze(1),g.unsqueeze(2)).view(-1) + 1e-8
            mult = F.relu(numer / denomin)
            a_old = action
            a_new = a_old - mult * g
            a_new = torch.clamp(a_new,-self.max_action, self.max_action)
            return a_new.cpu().data.numpy().flatten()
            
    def train(self, replay_buffer, batch_size=256):
        # Sample replay buffer 
        state, action, next_state, reward, cost, prev_cost, not_done = replay_buffer.sample(batch_size)

        # Compute the target C value
        target_C = cost

        # Get current C estimate
        current_C = self.C_critic(state, action) + prev_cost

        # Compute critic loss
        C_critic_loss = F.mse_loss(current_C, target_C)

        # Optimize the critic
        self.C_critic_optimizer.zero_grad()
        C_critic_loss.backward()
        self.C_critic_optimizer.step()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.rew_discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            action = self.actor(state)
            actor_loss = ( -self.critic.Q1(state, action)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
            for param, target_param in zip(self.C_critic.parameters(), self.C_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        
        
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.C_critic.state_dict(), filename + "_C_critic")
        torch.save(self.C_critic_optimizer.state_dict(), filename + "_C_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.C_critic.load_state_dict(torch.load(filename + "_C_critic"))
        self.C_critic_optimizer.load_state_dict(torch.load(filename + "_C_critic_optimizer"))
        self.C_critic_target = copy.deepcopy(self.C_critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, seed, eval_episodes=5,use_qpsl=False):
    avg_reward = 0.
    avg_cost = 0.
    for _ in range(eval_episodes):
        state, cost, done = eval_env.reset(), 0, False
        while not done:
            action = policy.select_action(np.array(state),prev_cost=cost, use_qpsl=use_qpsl)
            state, reward, done, info = eval_env.step(action)
            cost = 1 if info['cost']!=0 else 0
            avg_reward += reward
            avg_cost += cost

    avg_reward /= eval_episodes
    avg_cost /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} Cost {avg_cost:.3f}.\
            Use QP Safety Layer : {use_qpsl}")
    print("---------------------------------------")
    return avg_reward,avg_cost