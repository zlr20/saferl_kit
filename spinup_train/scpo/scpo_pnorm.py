import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import copy
import scpo_core as core
from utils.logx import EpochLogger, setup_logger_kwargs, colorize
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum
from safety_gym.envs.engine import Engine as safety_gym_Engine
from safety_gym_arm.envs.engine import Engine as safety_gym_arm_Engine
from utils.safetygym_config import configuration
import os.path as osp

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
INF = np.inf

class SCPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, cgamma=1., clam=0.95, pnorm=None):
        self.obs_buf      = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf      = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf      = np.zeros(size, dtype=np.float32)
        self.rew_buf      = np.zeros(size, dtype=np.float32)
        self.ret_buf      = np.zeros(size, dtype=np.float32)
        self.val_buf      = np.zeros(size, dtype=np.float32)
        self.cost_buf      = np.zeros(size, dtype=np.float32)
        self.cost_increase_buf     = np.zeros(size, dtype=np.float32)
        self.cost_increase_ret_buf = np.zeros(size, dtype=np.float32)
        self.cost_increase_val_buf = np.zeros(size, dtype=np.float32)
        self.adc_increase_buf      = np.zeros(size, dtype=np.float32)
        self.logp_buf     = np.zeros(size, dtype=np.float32)
        self.mu_buf       = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.logstd_buf   = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.cgamma, self.clam = cgamma, clam # there is no discount for the cost for MMDP 
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.pnorm = pnorm

    def store(self, obs, act, rew, val, logp, cost_increase, cost_increase_val, mu, logstd, cost):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr]      = obs
        self.act_buf[self.ptr]      = act
        self.rew_buf[self.ptr]      = rew
        self.val_buf[self.ptr]      = val
        self.logp_buf[self.ptr]     = logp
        self.cost_increase_buf[self.ptr]     = cost_increase
        self.cost_increase_val_buf[self.ptr] = cost_increase_val
        self.mu_buf[self.ptr]       = mu
        self.logstd_buf[self.ptr]   = logstd
        self.cost_buf[self.ptr] = cost
        self.ptr += 1

    def finish_path(self, last_val=0, last_cost_increase_val=0, pnorm=INF):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        costs_increase = np.append(self.cost_increase_buf[path_slice], last_cost_increase_val)
        cost_increase_vals = np.append(self.cost_increase_val_buf[path_slice], last_cost_increase_val)
        costs = np.append(self.cost_buf[path_slice], 0) # just assume that there is no cost
        past_maxs = self.obs_buf[path_slice,-1] # the last dimension of observation 
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # cost advantage calculation
        cost_increase_deltas = costs_increase[:-1] + self.cgamma * cost_increase_vals[1:] - cost_increase_vals[:-1]
        self.adc_increase_buf[path_slice] = core.discount_cumsum(cost_increase_deltas, self.cgamma * self.clam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        # costs_increase-to-go, targets for the cost value function
        # this needs to computed independently with p norm 
        # p infinity case 
        # if not self.pnorm:
        #     future_max_cost =core.future_max(costs)[:-1]
        # else:
        #     future_max_cost = core.future_max_norm(costs, pnorm=self.pnorm)[:-1]
            
        future_max_cost = core.future_max_norm(costs, pnorm=pnorm)[:-1]
            
        costs_increase_ret = [max(future_max_cost[i] - past_maxs[i], 0) for i in range(len(future_max_cost))]
        # self.cost_increase_ret_buf[path_slice] = core.discount_cumsum(costs_increase, self.cgamma)[:-1]
        self.cost_increase_ret_buf[path_slice] = costs_increase_ret
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        # center cost advantage, but don't scale
        adc_increase_mean, adc_increase_std = mpi_statistics_scalar(self.adc_increase_buf)
        self.adc_increase_buf = (self.adc_increase_buf - adc_increase_mean)
        data = dict(obs=torch.FloatTensor(self.obs_buf).to(device), 
                    act=torch.FloatTensor(self.act_buf).to(device), 
                    ret=torch.FloatTensor(self.ret_buf).to(device),
                    adv=torch.FloatTensor(self.adv_buf).to(device),
                    cost_increase_ret=torch.FloatTensor(self.cost_increase_ret_buf).to(device),
                    adc_increase=torch.FloatTensor(self.adc_increase_buf).to(device),
                    logp=torch.FloatTensor(self.logp_buf).to(device),
                    mu=torch.FloatTensor(self.mu_buf).to(device),
                    logstd=torch.FloatTensor(self.logstd_buf).to(device))
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


def get_net_param_np_vec(net):
    """
        Get the parameters of the network as numpy vector
    """
    return torch.cat([val.flatten() for val in net.parameters()], axis=0).detach().cpu().numpy()

def assign_net_param_from_flat(param_vec, net):
    param_sizes = [np.prod(list(val.shape)) for val in net.parameters()]
    ptr = 0
    for s, param in zip(param_sizes, net.parameters()):
        param.data.copy_(torch.from_numpy(param_vec[ptr:ptr+s]).reshape(param.shape))
        ptr += s

def cg(Ax, b, cg_iters=100):
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
        # early stopping 
        if np.linalg.norm(p) < EPS:
            break
    return x

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
    """
    Returns 
    """
    jacob = auto_grad(objective, net, to_numpy=False)
    
    return auto_grad(torch.dot(jacob, x), net, to_numpy=True)

def scpo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, vcf_lr=1e-3, train_v_iters=80, train_vc_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, target_cost = 1.5, logger_kwargs=dict(), save_freq=10, backtrack_coeff=0.8, 
        backtrack_iters=100, model_save=False, cost_reduction=0, pnorm=None):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn() 
    obs_dim = (env.observation_space.shape[0]+1,) # this is especially designed for SCPO, since we require an additional M in the observation space 
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = SCPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, pnorm=pnorm)
    
    def compute_kl_pi(data, cur_pi):
        """
        Return the sample average KL divergence between old and new policies
        """
        obs, act, adv, logp_old, mu_old, logstd_old = data['obs'], data['act'], data['adv'], data['logp'], data['mu'], data['logstd']
        
        # Average KL Divergence  
        pi, logp = cur_pi(obs, act)
        # average_kl = (logp_old - logp).mean()
        average_kl = cur_pi._d_kl(
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(mu_old, dtype=torch.float32),
            torch.as_tensor(logstd_old, dtype=torch.float32), device=device)
        
        return average_kl
    
    def compute_cost_pi(data, cur_pi):
        """
        Return the suggorate cost for current policy
        """
        obs, act, adc_increase, logp_old = data['obs'], data['act'], data['adc_increase'], data['logp']
        
        # Surrogate cost function 
        pi, logp = cur_pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        surr_cost = (ratio * adc_increase).sum()
        epochs = len(logger.epoch_dict['EpCost'])
        surr_cost /= epochs # the average 
        
        return surr_cost
        
        
    def compute_loss_pi(data, cur_pi):
        """
        The reward objective for SCPO (SCPO policy loss)
        """
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Policy loss 
        pi, logp = cur_pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)
        
        return loss_pi, pi_info
        
    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()
    
    # Set up function for computing cost loss 
    def compute_loss_vc(data):
        obs, cost_increase_ret = data['obs'], data['cost_increase_ret']
        return ((ac.vc(obs) - cost_increase_ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    vcf_optimizer = Adam(ac.vc.parameters(), lr=vcf_lr)

    # Set up model saving
    if model_save:
        logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # log the loss objective and cost function and value function for old policy
        pi_l_old, pi_info_old = compute_loss_pi(data, ac.pi)
        pi_l_old = pi_l_old.item()
        surr_cost_old = compute_cost_pi(data, ac.pi)
        surr_cost_old = surr_cost_old.item()
        v_l_old = compute_loss_v(data).item()


        # SCPO policy update core impelmentation 
        loss_pi, pi_info = compute_loss_pi(data, ac.pi)
        surr_cost = compute_cost_pi(data, ac.pi)
        
        # get Hessian for KL divergence
        kl_div = compute_kl_pi(data, ac.pi)
        Hx = lambda x: auto_hession_x(kl_div, ac.pi, torch.FloatTensor(x).to(device))
        
        # linearize the loss objective and cost function
        g = auto_grad(loss_pi, ac.pi) # get the loss flatten gradient evaluted at pi old 
        b = auto_grad(surr_cost, ac.pi) # get the cost flatten gradient evaluted at pi old
        
        # get the Episode cost
        EpLen = logger.get_stats('EpLen')[0]
        EpMaxCost = logger.get_stats('EpMaxCost')[0]
        
        # cost constraint linearization
        '''
        original fixed target cost, in the context of mean adv of epochs
        '''
        # c = EpMaxCost - target_cost 
        # rescale  = EpLen
        # c /= (rescale + EPS)
        
        '''
        fixed target cost, in the context of sum adv of epoch
        '''
        c = EpMaxCost - target_cost
        
        # core calculation for SCPO
        Hinv_g   = cg(Hx, g)             # Hinv_g = H \ g        
        approx_g = Hx(Hinv_g)           # g
        # q        = np.clip(Hinv_g.T @ approx_g, 0.0, None)  # g.T / H @ g
        q        = Hinv_g.T @ approx_g
        
        # solve QP
        # decide optimization cases (feas/infeas, recovery)
        # Determine optim_case (switch condition for calculation,
        # based on geometry of constrained optimization problem)
        if b.T @ b <= 1e-8 and c < 0:
            Hinv_b, r, s, A, B = 0, 0, 0, 0, 0
            optim_case = 4
        else:
            # cost grad is nonzero: SCPO update!
            Hinv_b = cg(Hx, b)                # H^{-1} b
            r = Hinv_b.T @ approx_g          # b^T H^{-1} g
            s = Hinv_b.T @ Hx(Hinv_b)        # b^T H^{-1} b
            A = q - r**2 / s            # should be always positive (Cauchy-Shwarz)
            B = 2*target_kl - c**2 / s  # does safety boundary intersect trust region? (positive = yes)

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
                print(colorize(f'Alert! Attempting feasible recovery!', 'yellow', bold=True))
            else:
                # x = 0 infeasible, and safety halfspace is outside trust region
                # ==> whole trust region is infeasible, try to fail gracefully
                optim_case = 0
                print(colorize(f'Alert! Attempting INFEASIBLE recovery!', 'red', bold=True))
        
        print(colorize(f'optim_case: {optim_case}', 'magenta', bold=True))
        
        
        # get optimal theta-theta_k direction
        if optim_case in [3,4]:
            lam = np.sqrt(q / (2*target_kl))
            nu = 0
        elif optim_case in [1,2]:
            LA, LB = [0, r /c], [r/c, np.inf]
            LA, LB = (LA, LB) if c < 0 else (LB, LA)
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(np.sqrt(A/B), LA)
            lam_b = proj(np.sqrt(q/(2*target_kl)), LB)
            f_a = lambda lam : -0.5 * (A / (lam+EPS) + B * lam) - r*c/(s+EPS)
            f_b = lambda lam : -0.5 * (q / (lam+EPS) + 2 * target_kl * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            # nu = max(0, lam * c - r) / (np.clip(s,0.,None)+EPS)
            nu = max(0, lam * c - r) / (s+EPS)
        else:
            lam = 0
            # nu = np.sqrt(2 * target_kl / (np.clip(s,0.,None)+EPS))
            nu = np.sqrt(2 * target_kl / (s+EPS))
            
        # normal step if optim_case > 0, but for optim_case =0,
        # perform infeasible recovery: step to purely decrease cost
        x_direction = (1./(lam+EPS)) * (Hinv_g + nu * Hinv_b) if optim_case > 0 else nu * Hinv_b
        
        # copy an actor to conduct line search 
        actor_tmp = copy.deepcopy(ac.pi)
        def set_and_eval(step):
            new_param = get_net_param_np_vec(ac.pi) - step * x_direction
            assign_net_param_from_flat(new_param, actor_tmp)
            kl = compute_kl_pi(data, actor_tmp)
            pi_l, _ = compute_loss_pi(data, actor_tmp)
            surr_cost = compute_cost_pi(data, actor_tmp)
            
            return kl, pi_l, surr_cost
        
        # update the policy such that the KL diveragence constraints are satisfied and loss is decreasing
        for j in range(backtrack_iters):
            try:
                kl, pi_l_new, surr_cost_new = set_and_eval(backtrack_coeff**j)
            except:
                import ipdb; ipdb.set_trace()
            
            if (kl.item() <= target_kl and
                (pi_l_new.item() <= pi_l_old if optim_case > 1 else True) and # if current policy is feasible (optim>1), must preserve pi loss
                # surr_cost_new - surr_cost_old <= max(-c,0)):
                surr_cost_new - surr_cost_old <= max(-c,-cost_reduction)):
                
                print(colorize(f'Accepting new params at step %d of line search.'%j, 'green', bold=False))
                
                # update the policy parameter 
                new_param = get_net_param_np_vec(ac.pi) - backtrack_coeff**j * x_direction
                assign_net_param_from_flat(new_param, ac.pi)
                
                loss_pi, pi_info = compute_loss_pi(data, ac.pi) # re-evaluate the pi_info for the new policy
                surr_cost = compute_cost_pi(data, ac.pi) # re-evaluate the surr_cost for the new policy
                break
            if j==backtrack_iters-1:
                print(colorize(f'Line search failed! Keeping old params.', 'yellow', bold=False))

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()
            
        # Cost value function learning
        for i in range(train_vc_iters):
            vcf_optimizer.zero_grad()
            loss_vc = compute_loss_vc(data)
            loss_vc.backward()
            mpi_avg_grads(ac.vc)    # average grads across MPI processes
            vcf_optimizer.step()

        # Log changes from update        
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old, LossCost=surr_cost_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old),
                     DeltaLossCost=(surr_cost.item() - surr_cost_old))

    # Prepare for interaction with environment
    start_time = time.time()
    
    while True:
        try:
            o, ep_ret, ep_len = env.reset(), 0, 0
            break
        except:
            print('reset environment is wrong, try next reset')
    ep_cost_ret, ep_cost = 0, 0
    cum_cost = 0
    M = 0. # initialize the current maximum cost
    o_aug = np.append(o, M) # augmented observation = observation + M 
    first_step = True

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, vc, logp, mu, logstd = ac.step(torch.as_tensor(o_aug, dtype=torch.float32))
            
            try: 
                next_o, r, d, info = env.step(a)
                assert 'cost' in info.keys()
            except: 
                # simulation exception discovered, discard this episode 
                next_o, r, d = o, 0, True # observation will not change, no reward when episode done 
                info['cost'] = 0 # no cost when episode done    
            
            if first_step:
                # the first step of each episode 
                cost_increase = info['cost'] # define the new observation and cost for Maximum Markov Decision Process
                M_next = info['cost']
                first_step = False
            else:
                # the second and forward step of each episode
                cost_increase = max(info['cost'] - M, 0) # define the new observation and cost for Maximum Markov Decision Process
                M_next = M + cost_increase
             
            # Track cumulative cost over training
            cum_cost += info['cost']
            ep_ret += r
            ep_cost_ret += info['cost'] * (gamma ** t)
            ep_cost += info['cost']
            ep_len += 1

            # save and log
            buf.store(o_aug, a, r, v, logp, cost_increase, vc, mu, logstd, info['cost'])
            logger.store(VVals=v)
            
            # Update obs (critical!)
            # o = next_o
            M = M_next
            o_aug = np.append(next_o, M_next)

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, vc, _, _, _ = ac.step(torch.as_tensor(o_aug, dtype=torch.float32))
                else:
                    v = 0
                    # vc = 0
                    _, _, vc, _, _, _ = ac.step(torch.as_tensor(o_aug, dtype=torch.float32)) # bootstrap, since the trajectory is cut off by the simulator
                buf.finish_path(v, vc, pnorm=pnorm*1.015**epoch)
                if terminal:
                    # only save EpRet / EpLen / EpCostRet if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCostRet=ep_cost_ret, EpCost=ep_cost, EpMaxCost=M)
                while True:
                    try:
                        o, ep_ret, ep_len = env.reset(), 0, 0
                        break
                    except:
                        print('reset environment is wrong, try next reset')
                ep_cost_ret, ep_cost = 0, 0
                M = 0. # initialize the current maximum cost 
                # o_aug = o.append(M) # augmented observation = observation + M 
                o_bug = np.append(o, M) # augmented observation = observation + M 
                first_step = True

        # Save model
        if ((epoch % save_freq == 0) or (epoch == epochs-1)) and model_save:
            logger.save_state({'env': env}, None)

        # Perform SCPO update!
        update()
        
        #=====================================================================#
        #  Cumulative cost calculations                                       #
        #=====================================================================#
        cumulative_cost = mpi_sum(cum_cost)
        cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('EpCostRet', average_only=True)
        logger.log_tabular('EpCost', average_only=True)
        logger.log_tabular('EpMaxCost', average_only=True)
        logger.log_tabular('CumulativeCost', cumulative_cost)
        logger.log_tabular('CostRate', cost_rate)
        logger.log_tabular('VVals', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('LossCost', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('DeltaLossCost', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        
        
def create_env(args):
    if 'My' not in args.task:
        env = safety_gym_arm_Engine(configuration(args.task, args))
    else:
        env = safety_gym_Engine(configuration(args.task, args))
    return env

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task', type=str, default='Mygoal4')
    parser.add_argument('--hazards_size', type=float, default=0.30)  # the default hazard size of safety gym 
    parser.add_argument('--target_cost', type=float, default=0.) # the cost limit for the environment
    parser.add_argument('--target_kl', type=float, default=0.02) # the kl divergence limit for SCPO
    parser.add_argument('--cost_reduction', type=float, default=0.) # the cost_reduction limit when current policy is infeasible
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--pnorm', type=float, default=INF)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='scpopnorm_schedule1.015')
    parser.add_argument('--model_save', action='store_true')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    
    exp_name = args.task + '_' + args.exp_name \
                + '_' + 'kl' + str(args.target_kl) \
                + '_' + 'target_cost' + str(args.target_cost) \
                + '_' + 'pnorm' + str(args.pnorm)
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    # whether to save model
    # model_save = True if args.model_save else False
    model_save = True

    scpo(lambda : create_env(args), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, target_cost=args.target_cost, 
        model_save=model_save, target_kl=args.target_kl, cost_reduction=args.cost_reduction, pnorm=args.pnorm)