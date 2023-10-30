import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import a2c_core as core
from utils.logx import EpochLogger, setup_logger_kwargs
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from safe_rl_envs.envs.engine import Engine as  safe_rl_envs_Engine
from utils.safe_rl_env_config import configuration
import os.path as osp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class A2CBuffer:
    """
    A buffer for storing trajectories experienced by a A2C agent interacting
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
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
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
        # data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
        #             adv=self.adv_buf, logp=self.logp_buf)
        data = dict(obs=torch.FloatTensor(self.obs_buf).to(device), 
                    act=torch.FloatTensor(self.act_buf).to(device), 
                    ret=torch.FloatTensor(self.ret_buf).to(device),
                    adv=torch.FloatTensor(self.adv_buf).to(device), 
                    logp=torch.FloatTensor(self.logp_buf).to(device))
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def a2c(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(),  seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, delay=2, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10, atari=None):
    """
    Vanilla Policy Gradient 

    (with GAE-Lambda for advantage estimation)

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
            you provided to A2C.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

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
    buf = A2CBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing A2C policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        loss_pi = -(logp * adv).mean()

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
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with a single step of gradient descent
        # for i in range(max(1, train_v_iters // delay)):
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        mpi_avg_grads(ac.pi)    # average grads across MPI processes
        pi_optimizer.step()

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
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    if atari == None:
        o, ep_ret, ep_len = env.reset(), 0, 0
    else:
        o, ep_ret, ep_len = env.reset(seed=seed), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            if isinstance(o, tuple):
                o = o[0]
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            if atari == None:
                next_o, r, d, info = env.step(a)
            else:
                next_o, r, d_1, d_2, info = env.step(a)
                d = d_1 or d_2
            ep_ret += r
            ep_len += 1

            # save and log
            if isinstance(o, tuple):
                o = o[0]
            buf.store(o, a, r, v, logp)
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
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                if atari == None:
                    o, ep_ret, ep_len = env.reset(), 0, 0
                else:
                    o, ep_ret, ep_len = env.reset(seed=seed), 0, 0


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform A2C update!
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
        logger.dump_tabular()

def create_env(args):
    env = safe_rl_envs_Engine(configuration(args.task))
    return env

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Mygoal4')
    parser.add_argument('--hazards_size', type=float, default=0.30)  # the default hazard size of safety gym 
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--delay', type=int, default=2)
    parser.add_argument('--train_v_iters', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='a2c')
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
                                 'Tennis', 'Tutankham', 'WizardOfWor', 'YarsRevenge',
                                 
                                 'DonkeyKong', 'Darkchambers', 'Atlantis2', 'Backgammon', 'BasicMath', 'Blackjack', 'Casino',
                                 'Crossbow', 'Earthworld', 'Entombed', 'Et', 'FlagCapture', 'Frogger', 'Galaxian',
                                 'Hangman', 'HauntedHouse', 'HumanCannonball', 'Kaboom', 'KeystoneKapers', 'KingKong',
                                 'Klax', 'Koolaid', 'LaserGates', 'LostLuggage', 'MarioBros', 'MiniatureGolf',
                                 'MrDo', 'Othello', 'Pacman', 'Pitfall2', 'SirLancelot', 'SpaceWar',
                                 'Superman', 'Surround', 'Tetris', 'TicTacToe3D', 'Trondead', 'Turmoil',
                                 'VideoCheckers', 'VideoChess', 'VideoCube', 'WordZapper'])
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    if args.atari_name == None:
        exp_name = args.task + '_' + args.exp_name + '_' + 'epochs' + str(args.epochs)
    else:
        import gymnasium
        exp_name = args.atari_name + '_' + args.exp_name + '_' + 'epochs' + str(args.epochs)
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    a2c(lambda : create_env(args), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, atari=args.atari_name)