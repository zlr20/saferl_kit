import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import copy
import trpoissa_core as core
from utils.logx import EpochLogger, setup_logger_kwargs, colorize
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum
from safe_rl_envs.envs.engine import Engine as  safe_rl_envs_Engine
from utils.safe_rl_env_config import configuration
import os.path as osp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

class ISSABuffer:
    """
    A buffer for storing trajectories experienced by a ISSA agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.mu_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.logstd_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, mu, logstd):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.mu_buf[self.ptr] = mu
        self.logstd_buf[self.ptr] = logstd
        self.ptr += 1

    def finish_path(self, last_val=0):
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
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
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
        data = dict(obs=torch.FloatTensor(self.obs_buf).to(device), 
                    act=torch.FloatTensor(self.act_buf).to(device), 
                    ret=torch.FloatTensor(self.ret_buf).to(device),
                    adv=torch.FloatTensor(self.adv_buf).to(device), 
                    logp=torch.FloatTensor(self.logp_buf).to(device),
                    mu=torch.FloatTensor(self.mu_buf).to(device),
                    logstd=torch.FloatTensor(self.logstd_buf).to(device),
        )
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

def trpoissa(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, 
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, backtrack_coeff=0.8, backtrack_iters=100, model_save=False,
        adaptive_k=1, adaptive_n=1, adaptive_sigma=0.04):
    """
    Implicit Safe Set Algorithm (by using TRPO) 
 
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

        vf_lr (float): Learning rate for value function optimizer.

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
            
        backtrack_coeff (float): Scaling factor for line search.
        
        backtrack_iters (int): Number of line search steps.
        
        model_save (bool): If saving model.
        
        adaptive_k (int): hyperparameter of safety index.
        
        adaptive_n (int): hyperparameter of safety index.
        
        adaptive_sigma (float): hyperparameter of safety index.
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
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = ISSABuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    local_steps_per_epoch_eval = 10000

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


    def compute_loss_pi(data, cur_pi):
        """
        The reward objective for ISSA (ISSA policy loss)
        """
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Policy loss 
        pi, logp = cur_pi(obs, act)
        # loss_pi = -(logp * adv).mean()
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

    # Set up optimizers for policy and value function
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    if model_save:
        logger.setup_pytorch_saver(ac)

    def chk_unsafe(s, point, dt_ratio, dt_adamba, env, threshold, margin, adaptive_k, adaptive_n, adaptive_sigma, trigger_by_pre_execute, pre_execute_coef):
        action = point.tolist()
        # save state of env
        stored_state = copy.deepcopy(env.sim.get_state())
        safe_index_now = env.adaptive_safety_index(k=adaptive_k, sigma=adaptive_sigma, n=adaptive_n)

        # simulate the action
        try:
            s_new,_,_,info = env.step(action, simulate_in_adamba=True)
            if 'cost_exception' not in info:    
                safe_index_future = env.adaptive_safety_index(k=adaptive_k, sigma=adaptive_sigma, n=adaptive_n)
                if safe_index_future < max(0, safe_index_now):
                    flag = 0  # safe
                else:
                    flag = 1  # unsafe
            else:
                flag = -1
        except: 
            flag = -1  # expection
            
        if flag == -1:
            while True:
                try:
                    o = env.reset()
                    break
                except:
                    print('reset environment is wrong, try next reset')
        # set qpos and qvel
        env.sim.set_state(stored_state)
    
        # Note that the position-dependent stages of the computation must have been executed for the current state in order for these functions to return correct results. So to be safe, do mj_forward and then mj_jac. If you do mj_step and then call mj_jac, the Jacobians will correspond to the state before the integration of positions and velocities took place.
        env.sim.forward()
        return flag, env

    def outofbound(action_limit, action):
        flag = False
        for i in range(len(action_limit)):
            assert action_limit[i][1] > action_limit[i][0]
            if action[i] < action_limit[i][0] or action[i] > action_limit[i][1]:
                flag = True
                break
        return flag

    # Correction function of ISSA using AdamBA
    def AdamBA_SC(s, u, env, threshold=0, dt_ratio=1.0, ctrlrange=10.0, margin=0.4, adaptive_k=3, adaptive_n=1, adaptive_sigma=0.04, trigger_by_pre_execute=False, pre_execute_coef=0.0, vec_num=None, max_trial_num =1):
        infSet = []

        u = np.clip(u, -ctrlrange, ctrlrange)

        #action_space_num = 2
        action_space_num = env.action_space.shape[0]
        action = np.array(u).reshape(-1, action_space_num)

        dt_adamba = env.model.opt.timestep * env.frameskip_binom_n * dt_ratio

        assert dt_ratio == 1

        limits= [[-ctrlrange, ctrlrange]] * action_space_num
        NP = action

        # generate direction
        NP_vec_dir = []
        NP_vec = []

        loc = 0 
        scale = 0.1
        
        # num of actions input, default as 1
        for t in range(0, NP.shape[0]):
            if action_space_num == 2:
                vec_set = []
                vec_dir_set = []
                for m in range(0, vec_num):
                    theta_m = m * (2 * np.pi / vec_num)
                    vec_dir = np.array([np.sin(theta_m), np.cos(theta_m)]) / 2
                    vec_dir_set.append(vec_dir)
                    vec = NP[t]
                    vec_set.append(vec)
                NP_vec_dir.append(vec_dir_set)
                NP_vec.append(vec_set)
            else:
                vec_dir_set = np.random.normal(loc=loc, scale=scale, size=[vec_num, action_space_num])
                vec_set = [NP[t]] * vec_num
                NP_vec_dir.append(vec_dir_set)
                NP_vec.append(vec_set)

        bound = 0.0001

        # record how many boundary points have been found
        valid = 0
        cnt = 0
        out = 0
        yes = 0
        
        max_trials = max_trial_num
        for n in range(0, NP.shape[0]):
            trial_num = 0
            at_least_1 = False
            while trial_num < max_trials and not at_least_1:
                at_least_1 = False
                trial_num += 1
                NP_vec_tmp = copy.deepcopy(NP_vec[n])

                if trial_num ==1:
                    NP_vec_dir_tmp = NP_vec_dir[n]
                else:
                    NP_vec_dir_tmp = np.random.normal(loc=loc, scale=scale, size=[vec_num, action_space_num])

                for v in range(0, vec_num):
                    NP_vec_tmp_i = NP_vec_tmp[v]

                    NP_vec_dir_tmp_i = NP_vec_dir_tmp[v]

                    eta = bound
                    decrease_flag = 0
                    
                    opt_solution = []
                    while True: 
                        
                        flag, env = chk_unsafe(s, NP_vec_tmp_i, dt_ratio=dt_ratio, dt_adamba=dt_adamba, env=env,
                                            threshold=threshold, margin=margin, adaptive_k=adaptive_k, adaptive_n=adaptive_n, adaptive_sigma=adaptive_sigma,
                                            trigger_by_pre_execute=trigger_by_pre_execute, pre_execute_coef=pre_execute_coef)

                        # safety gym env itself has clip operation inside
                        if outofbound(limits, NP_vec_tmp_i):
                            break

                        if flag == -1:
                            # simulation expection
                            break
                        
                        if eta <= bound and decrease_flag == 1:
                            NP_vec_tmp_i = opt_solution
                            at_least_1 = True
                            break

                        # AdamBA procudure
                        if flag == 1 and decrease_flag == 0:
                            # outreach
                            NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                            eta = eta * 2
                            continue
                        
                        # monitor for 1st reaching out boundary
                        if flag == 0 and decrease_flag == 0:
                            decrease_flag = 1
                            opt_solution = NP_vec_tmp_i
                            eta = eta * 0.25  # make sure decrease step start at 0.5 of last increasing step
                            continue
                        # decrease eta
                        if flag == 1 and decrease_flag == 1:
                            NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                            eta = eta * 0.5
                            continue
                        if flag == 0 and decrease_flag == 1:
                            opt_solution = NP_vec_tmp_i
                            NP_vec_tmp_i = NP_vec_tmp_i - eta * NP_vec_dir_tmp_i
                            eta = eta * 0.5
                            continue

                    NP_vec_tmp[v] = NP_vec_tmp_i

            NP_vec_tmp_new = []
            for vnum in range(0, len(NP_vec_tmp)):
                cnt += 1
                if outofbound(limits, NP_vec_tmp[vnum]):
                    out += 1
                    continue
                if NP_vec_tmp[vnum][0] == u[0] and NP_vec_tmp[vnum][1] == u[1]:
                    yes += 1
                    continue

                valid += 1
                NP_vec_tmp_new.append(NP_vec_tmp[vnum])
            NP_vec[n] = NP_vec_tmp_new

        NP_vec_tmp = NP_vec[0]

        if valid > 0:
            valid_adamba_sc = "adamba_sc success"
        elif valid == 0 and yes==vec_num:
            valid_adamba_sc = "itself satisfy"
        elif valid == 0 and out==vec_num:
            valid_adamba_sc = "all out"
        else:
            valid_adamba_sc = "exception"
            print("out = ", out)
            print("yes = ", yes)
            print("valid = ", valid)

        if len(NP_vec_tmp) > 0:  # at least we have one sampled action satisfying the safety index 
            norm_list = np.linalg.norm(NP_vec_tmp, axis=1)
            optimal_action_index = np.where(norm_list == np.amin(norm_list))[0][0]
            return NP_vec_tmp[optimal_action_index], valid_adamba_sc, env, NP_vec_tmp
        else:
            return None, valid_adamba_sc, env, None
    
    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data, ac.pi)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()


        # ISSA policy update core impelmentation 
        loss_pi, pi_info = compute_loss_pi(data, ac.pi)
        g = auto_grad(loss_pi, ac.pi) # get the flatten gradient evaluted at pi old 
        kl_div = compute_kl_pi(data, ac.pi)
        Hx = lambda x: auto_hession_x(kl_div, ac.pi, torch.FloatTensor(x).to(device))
        x_hat    = cg(Hx, g)             # Hinv_g = H \ g
        
        s = x_hat.T @ Hx(x_hat)
        s_ep = s if s < 0. else 1 # log s negative appearence 
            
        x_direction = np.sqrt(2 * target_kl / (s+EPS)) * x_hat
        
        # copy an actor to conduct line search 
        actor_tmp = copy.deepcopy(ac.pi)
        def set_and_eval(step):
            new_param = get_net_param_np_vec(ac.pi) - step * x_direction
            assign_net_param_from_flat(new_param, actor_tmp)
            kl = compute_kl_pi(data, actor_tmp)
            pi_l, _ = compute_loss_pi(data, actor_tmp)
            
            return kl, pi_l
        
        # update the policy such that the KL diveragence constraints are satisfied and loss is decreasing
        for j in range(backtrack_iters):
            try:
                kl, pi_l_new = set_and_eval(backtrack_coeff**j)
            except:
                # import ipdb; ipdb.set_trace()
                break
            
            if (kl.item() <= target_kl and pi_l_new.item() <= pi_l_old):
                print(colorize(f'Accepting new params at step %d of line search.'%j, 'green', bold=False))
                # update the policy parameter 
                new_param = get_net_param_np_vec(ac.pi) - backtrack_coeff**j * x_direction
                assign_net_param_from_flat(new_param, ac.pi)
                loss_pi, pi_info = compute_loss_pi(data, ac.pi) # re-evaluate the pi_info for the new policy
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

        # Log changes from update        
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old),
                     EpochS = s_ep)

    # Prepare for interaction with environment
    start_time = time.time()
    while True:
        try:
            o, ep_ret, ep_len = env.reset(), 0, 0
            break
        except:
            print('reset environment is wrong, try next reset')
    ep_cost, cum_cost, ep_cost_issa = 0, 0, 0
    cum_cost_eval = 0
    cum_cost_issa = 0
    AdamBA_cnt = 0
    ISSA_cnt = 0
    
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        EP_start_time=time.time()
        for t in range(local_steps_per_epoch):
            if t % 1000 == 0:
                AdamBA_cnt = 0
            a, v, logp, mu, logstd = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    
            a_safe, valid_adamba_sc, _, _ = AdamBA_SC(o, a, env, vec_num=5, trigger_by_pre_execute=True, adaptive_k=adaptive_k, adaptive_n=adaptive_n, adaptive_sigma=adaptive_sigma)
            if a_safe is None:
                a_safe = a
            if a_safe is not a:
                AdamBA_cnt += 1
                ISSA_cnt += 1
                    
            try: 
                next_o, r, d, info = env.step(a_safe)
                assert 'cost' in info.keys()
            except: 
                # simulation exception discovered, discard this episode 
                next_o, r, d = o, 0, True # observation will not change, no reward when episode done 
                info['cost'] = 0 # no cost when episode done 
            
            # Track cumulative cost over training
            cum_cost += info['cost']
            
            ep_ret += r
            ep_len += 1
            ep_cost += info['cost']

            # save and log
            buf.store(o, a, r, v, logp, mu, logstd)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen / EpCost if trajectory finished
                    logger.store(EpRet_train=ep_ret, EpLen_train=ep_len, EpCost_train=ep_cost, EpCost_ISSA=ep_cost_issa, EpISSA_train=ISSA_cnt,EPTime_train=time.time()-EP_start_time)
                while True:
                    try:
                        o, ep_ret, ep_len = env.reset(), 0, 0
                        break
                    except:
                        print('reset environment is wrong, try next reset')
                ep_cost = 0 # episode cost is zero 
                ep_cost_issa = 0
                ISSA_cnt = 0
                EP_start_time = time.time()

        ###########################################################################################       
        # evaluate without ISSA  
        while True:
            try:
                o, ep_ret_eval, ep_len_eval = env.reset(), 0, 0
                break
            except:
                print('reset environment is wrong, try next reset')
        ep_cost_eval = 0
        
        EP_start_time_eval=time.time()
        for t in range(local_steps_per_epoch_eval):
            a, v, logp, mu, logstd = ac.step(torch.as_tensor(o, dtype=torch.float32))
            a_safe = a 
                    
            try: 
                next_o, r, d, info = env.step(a_safe)
                assert 'cost' in info.keys()
            except: 
                # simulation exception discovered, discard this episode 
                next_o, r, d = o, 0, True # observation will not change, no reward when episode done 
                info['cost'] = 0 # no cost when episode done 
            
            # Track cumulative cost over training
            cum_cost_eval += info['cost']
            
            ep_ret_eval += r
            ep_len_eval += 1
            ep_cost_eval += info['cost']

            logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len_eval == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch_eval-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                # buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen / EpCost if trajectory finished
                    logger.store(EpRet=ep_ret_eval, EpLen=ep_len_eval, EpCost=ep_cost_eval,EPTime=time.time()-EP_start_time_eval)
                while True:
                    try:
                        o, ep_ret_eval, ep_len_eval = env.reset(), 0, 0
                        break
                    except:
                        print('reset environment is wrong, try next reset')
                ep_cost_eval = 0 # episode cost is zero 
                EP_start_time_eval = time.time()
        
        
        # Save model
        if ((epoch % save_freq == 0) or (epoch == epochs-1)) and model_save:
            logger.save_state({'env': env}, None)

        # Perform ISSA update!
        update()
        
        #=====================================================================#
        #  Cumulative cost calculations                                       #
        #=====================================================================#
        cumulative_cost = mpi_sum(cum_cost)
        cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)

        cumulative_cost_eval = mpi_sum(cum_cost_eval)
        cost_rate_eval = cumulative_cost_eval / ((epoch+1)*local_steps_per_epoch_eval)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpCost', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('CumulativeCost', cumulative_cost_eval)
        logger.log_tabular('CostRate', cost_rate_eval)
        
        logger.log_tabular('EpRet_train', average_only=True)
        logger.log_tabular('EpCost_train', average_only=True)
        logger.log_tabular('EpCost_ISSA', average_only=True)
        logger.log_tabular('EpLen_train', average_only=True)
        logger.log_tabular('EpISSA_train', average_only=True)
        logger.log_tabular('EPTime_train', average_only=True)
        logger.log_tabular('EPTime', average_only=True)
        logger.log_tabular('CumulativeCost_train', cumulative_cost)
        logger.log_tabular('CostRate_train', cost_rate)
        
        logger.log_tabular('VVals', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.log_tabular('EpochS', average_only=True)
        logger.dump_tabular()
        
        
def create_env(args):
    env = safe_rl_envs_Engine(configuration(args.task))
    return env

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task', type=str, default='Goal_Point_8Hazards')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='trpoissa')
    parser.add_argument('--model_save', action='store_true')
    parser.add_argument('--target_kl', type=float, default=0.02)
    parser.add_argument('--adaptive_k', type=float, default=1)
    parser.add_argument('--adaptive_n', type=float, default=1)
    parser.add_argument('--adaptive_sigma', type=float, default=0.04)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    
    # exp_name = args.task + '_' + args.exp_name + '_' + 'kl' + str(args.target_kl) + '_' + 'epochs' + str(args.epochs)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # whether to save model
    model_save = True if args.model_save else False

    trpoissa(lambda : create_env(args), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, model_save=model_save, target_kl=args.target_kl, max_ep_len=args.max_ep_len,
        adaptive_k = args.adaptive_k, adaptive_n = args.adaptive_n, adaptive_sigma = args.adaptive_sigma)