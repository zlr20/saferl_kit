import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import copy
import apo_core as core
from utils.logx import EpochLogger, setup_logger_kwargs, colorize
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum
from safe_rl_envs.envs.engine import Engine as  safe_rl_envs_Engine
from utils.safe_rl_env_config import configuration
import os.path as osp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

class APOBuffer:
    """
    A buffer for storing trajectories experienced by a APO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, act_n=1):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.discounted_adv_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.mu_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.logstd_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.logits_buf = np.zeros(core.combined_shape(size, act_n), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, mu=np.nan, logstd=np.nan, logits=np.nan):
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
        self.logits_buf[self.ptr] = logits
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
        self.discounted_adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        #advantage of every (s, a) pair
        self.adv_buf[path_slice] = deltas
        
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
        self.ptr, self.path_start_idx = 0, 0    # reset the buffer
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.discounted_adv_buf)
        self.discounted_adv_buf = (self.discounted_adv_buf - adv_mean) / adv_std
        data = dict(obs=torch.FloatTensor(self.obs_buf).to(device), 
                    act=torch.FloatTensor(self.act_buf).to(device), 
                    ret=torch.FloatTensor(self.ret_buf).to(device),
                    disc_adv=torch.FloatTensor(self.discounted_adv_buf).to(device), 
                    adv=torch.FloatTensor(self.adv_buf).to(device), 
                    logp=torch.FloatTensor(self.logp_buf).to(device),
                    mu=torch.FloatTensor(self.mu_buf).to(device),
                    logstd=torch.FloatTensor(self.logstd_buf).to(device),
                    logits=torch.FloatTensor(self.logits_buf).to(device),
                    val=torch.FloatTensor(self.val_buf).to(device),
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

def apo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, 
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, backtrack_coeff=0.8, backtrack_iters=100, model_save=False, 
        k=10., omega_1=0.01, omega_2=0.01, atari=None, detailed=False):
    """
    Absolute Policy Optimization
 
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
            you provided to APO.

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
        
        k (int): Probability Factor.
        
        omega_1 (float): hyperparameter for the infinite norm of mu.
        
        omega_2 (float): hyperparameter for H_max. 
        
        atari (str): name of atari game (None if running continuous game).
        
        detailed (bool): whether to display detailed computation of square item in variance mean

    """
    
    def atari_env_fn(atari_name, version='5'):
        env_name = 'ALE/' + atari_name + '-v' + version
        return gymnasium.make(env_name, obs_type="ram")

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
    if atari == None:
        env = env_fn()
    else:
        env = atari_env_fn(atari)
    obs_dim = env.observation_space.shape
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
    if atari == None:
        buf = APOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    else:
        buf = APOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, env.action_space.n)

    def compute_kl_pi(data, cur_pi):
        """
        Return the sample average KL divergence between old and new policies
        """
        obs = data['obs']
    
        if atari == None:
            mu_old, logstd_old =  data['mu'], data['logstd']
            average_kl = cur_pi._d_kl(
                torch.as_tensor(obs, dtype=torch.float32),
                torch.as_tensor(mu_old, dtype=torch.float32),
                torch.as_tensor(logstd_old, dtype=torch.float32), device=device)
        else:
            logits_old = data['logits']
            average_kl = cur_pi._d_kl(
                torch.as_tensor(obs, dtype=torch.float32),
                torch.as_tensor(logits_old, dtype=torch.float32), device=device)
        
        return average_kl
    
    def compute_loss_pi_Chebyshev(data, cur_pi):
        """
        The reward objective APO (APO policy loss)
        """
        obs, act, disc_adv, adv, logp_old, val = data['obs'], data['act'], data['disc_adv'], data['adv'], data['logp'], data['val']
        pi, logp = cur_pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        
        mean_surr = (ratio*disc_adv).mean()
        
        tmp_1 = (ratio-1)*adv**2
        tmp_2 = 2*ratio*adv
        mean_var_surr = omega_1 * abs(tmp_1+tmp_2*omega_2).mean()
        
        if detailed:
            kl_div = abs((logp_old - logp).mean().item())
            epsilon = max(disc_adv)
            bias = 4*gamma*kl_div*epsilon/(1-gamma)**2
            min_J_square = mean_surr**2 + 2*val.mean()*mean_surr
            if mean_surr + val.mean() - bias < 0:
                min_J_square = 0
        else:
            min_J_square = mean_surr**2 + 2*val.mean()*mean_surr

        factor = omega_1 * (1 - gamma**2) / k
        L_ = abs(disc_adv)
        var_mean_surr = factor * (L_**2 + 2*L_*val).mean() - min_J_square
        
        # loss 
        loss_pi = -(mean_surr - k*(mean_var_surr + var_mean_surr))*2/3.0 - mean_surr/3.0
        
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

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi_Chebyshev(data, ac.pi)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # APO policy update core impelmentation 
        loss_pi, pi_info = compute_loss_pi_Chebyshev(data, ac.pi)
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
            pi_l, _ = compute_loss_pi_Chebyshev(data, actor_tmp)
            
            return kl, pi_l
        
        # update the policy such that the KL diveragence constraints are satisfied and loss is decreasing
        for j in range(backtrack_iters):
            try:
                kl, pi_l_new = set_and_eval(backtrack_coeff**j)
            except:
                import ipdb; ipdb.set_trace()
            
            if (kl.item() <= target_kl and pi_l_new.item() <= pi_l_old):
                print(colorize(f'Accepting new params at step %d of line search.'%j, 'green', bold=False))
                # update the policy parameter 
                new_param = get_net_param_np_vec(ac.pi) - backtrack_coeff**j * x_direction
                assign_net_param_from_flat(new_param, ac.pi)
                loss_pi, pi_info = compute_loss_pi_Chebyshev(data, ac.pi) # re-evaluate the pi_info for the new policy
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
    # o, ep_ret, ep_len = env.reset(), 0, 0
    while True:
        try:
            if atari == None:
                o, ep_ret, ep_len = env.reset(), 0, 0
            else:
                o, ep_ret, ep_len = env.reset(seed=seed), 0, 0
            break
        except:
            print('reset environment is wrong, try next reset')
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            if isinstance(o, tuple):
                o = o[0]
            if atari == None:
                a, v, logp, mu, logstd = ac.step(torch.as_tensor(o, dtype=torch.float32))
            else:
                a, v, logp, logits = ac.step(torch.as_tensor(o, dtype=torch.float32))
            
            try:
                if atari == None:
                    next_o, r, d, info = env.step(a)
                else:
                    next_o, r, d_1, d_2, info = env.step(a)
                    d = d_1 or d_2
            except: 
                print(f"simulation exception discovered, discard this episode")
                # simulation exception discovered, discard this episode 
                next_o, r, d = o, 0, True # observation will not change, no reward when episode done 
            
            ep_ret += r
            ep_len += 1

            # save and log
            if isinstance(o, tuple):
                o = o[0]
            if atari == None:
                buf.store(o, a, r, v, logp, mu=mu, logstd=logstd)
            else:
                buf.store(o, a, r, v, logp, logits=logits)
            logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o

            atari_mode = atari != None
            timeout = (ep_len == max_ep_len) and (not atari_mode)
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    if isinstance(o, tuple):
                        o = o[0]
                    if atari == None:
                        _, v, _, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        _, v, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                while True:
                    try:
                        if atari == None:
                            o, ep_ret, ep_len = env.reset(), 0, 0
                        else:
                            o, ep_ret, ep_len = env.reset(seed=seed), 0, 0
                        break
                    except:
                        print('reset environment is wrong, try next reset')

        # Save model
        if ((epoch % save_freq == 0) or (epoch == epochs-1)) and model_save:
            logger.save_state({'env': env}, None)

        # Perform APO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
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
    parser.add_argument('--task', type=str, default='Goal_Point')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200)          
    parser.add_argument('--exp_name', type=str, default='apo')
    parser.add_argument('--model_save', action='store_true')
    parser.add_argument('--target_kl', type=float, default=0.02)    
    parser.add_argument('--omega1', type=float, default=0.001)       
    parser.add_argument('--omega2', type=float, default=0.005)       
    parser.add_argument('--k', '-k', type=float, default=10.5)
    parser.add_argument('--detailed', '-d', action='store_true', default=False)        
    parser.add_argument('--atari_name', '-a', type=str, default=None, 
                        choices=['Adventure', 'Pong', 'Seaquest', 'Riverraid', 'Freeway', 'BeamRider', 'Gopher', 'SpaceInvaders',
                                 'AirRaid', 'Assault', 'Qbert', 'Skiing', 'Enduro', 'Breakout', 'Bowling', 'IceHockey', 'KungFuMaster',
                                 'TimePilot', 'Boxing', 'JourneyEscape', 'CrazyClimber', 'Frostbite',
                                 'Asteroids', 'Solaris', 'Zaxxon', 'BattleZone', 'Centipede', 'DemonAttack',
                                 'StarGunner', 'VideoPinball', 'Venture', 'UpNDown', 'Robotank',
                                 'Atlantis', 'Carnival', 'Defender', 'ElevatorAction', 'Hero',
                                 'Pitfall', 'Amidar', 'FishingDerby', 'MsPacman',
                                 'Alien', 'Asterix', 'BankHeist', 'Berzerk', 'ChopperCommand',
                                 'DoubleDunk', 'Gravitar', 'Jamesbond', 'Kangaroo', 'NameThisGame',
                                 'Krull', 'MontezumaRevenge', 'Phoenix', 'Pooyan', 'PrivateEye', 'RoadRunner',
                                 'Tennis', 'Tutankham', 'WizardOfWor', 'YarsRevenge'])
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    
    if args.atari_name == None:
        exp_name = args.task + '_' + args.exp_name + '_' + 'kl' + str(args.target_kl) + '_' + 'epochs' + str(args.epochs)
    else:
        import gymnasium
        exp_name = args.atari_name + '_' + args.exp_name + '_' + 'kl' + str(args.target_kl) + '_' + 'epochs' + str(args.epochs)
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    # whether to save model
    model_save = True if args.model_save else False

    apo(lambda : create_env(args), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, model_save=model_save, target_kl=args.target_kl, max_ep_len=args.max_ep_len,
        k=args.k, omega_1=args.omega1, omega_2=args.omega2, atari=args.atari_name, detailed=args.detailed)
    

