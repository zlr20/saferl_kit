import os,sys,argparse,warnings
warnings.filterwarnings("ignore")
import numpy as np
import gym
import torch
import saferl_algos
from saferl_plotter.logger import SafeLogger
import saferl_utils

######################### make cartpole stabilization environment ################################################################
sys.path.append("saferl_envs")
from copy import deepcopy
from functools import partial
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, set_device_from_config, set_seed_from_config, save_video

def create_env(seed=None,device= "cuda" if torch.cuda.is_available() else "cpu"):
    fac = ConfigFactory()
    config = fac.merge()
    config.seed = seed
    config.device = device
    set_seed_from_config(config)
    set_device_from_config(config)
    env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
    return env_func()
##################################################################################################################################


# python main.py --task cartpole --overrides saferl_envs/config_overrides/stable_task.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--overrides",nargs='+',type=str)
    parser.add_argument("--exp_name",type=str)
    parser.add_argument("--env", default="Stabilization")           # Env name
    parser.add_argument("--flag", default="constraint_violation")   # c_t = info[flag]
    parser.add_argument("--base_policy", default="TD3")             # Base Policy name
    parser.add_argument("--use_td3", action="store_true")           # unconstrained RL
    parser.add_argument("--use_usl", action="store_true")           # Wether to use Unrolling Safety Layer
    parser.add_argument("--use_qpsl",action="store_true")           # Wether to use QP Safety Layer (Dalal 2018)
    parser.add_argument("--use_recovery",action="store_true")       # Wether to use Recovery RL     (Thananjeyan 2021)
    parser.add_argument("--use_lag",action="store_true")            # Wether to use Lagrangian Relaxation  (Ha 2021)
    parser.add_argument("--use_fac",action="store_true")            # Wether to use FAC (Ma 2021)
    parser.add_argument("--use_rs",action="store_true")             # Wether to use Reward Shaping
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    # Hyper-parameters for all safety-aware algorithms
    parser.add_argument("--delta",default = 0.1,type=float)         # Qc(s,a) \leq \delta
    parser.add_argument("--cost_discount", default=0.99)            # Discount factor for cost-return
    # Hyper-parameters for using USL
    parser.add_argument("--warmup_ratio", default=1/3)              # Start using USL in traing after max_timesteps*warmup_ratio steps
    parser.add_argument("--kappa",default = 5, type=float)                      # Penalized factor for Stage 1
    parser.add_argument("--early_stopping", action="store_true")    # Wether to terminate an episode upon cost > 0
    # Hyper-parameters for using Lagrangain Relaxation
    parser.add_argument("--lam_init", default = 0.)                 # Initalize lagrangian multiplier
    parser.add_argument("--lam_lr",default = 1e-5)                  # Step-size of multiplier update
    # Hyper-parameters for using Reward Shaping
    parser.add_argument("--cost_penalty",default = 0.1)               # Step-size of multiplier update
    # Other hyper-parameters for original TD3
    parser.add_argument("--start_timesteps", default=1000, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=1000, type=int)      # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--rew_discount", default=0.99)             # Discount factor for reward-return
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    assert [bool(i) for i in [args.use_td3,args.use_usl,args.use_recovery,args.use_qpsl,args.use_lag,args.use_fac,args.use_rs]].count(True) == 1, 'Only one option can be True'


    if not args.exp_name:
        if args.use_usl:
            file_name = f"{'usl'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='1_USL',env_name=args.env,seed=args.seed,fieldnames=['EpRet','EpCost','CostRate'])
        elif args.use_recovery:
            file_name = f"{'rec'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='2_REC',env_name=args.env,seed=args.seed,fieldnames=['EpRet','EpCost','CostRate'])
        elif args.use_qpsl:
            file_name = f"{'qpsl'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='3_QPSL',env_name=args.env,seed=args.seed,fieldnames=['EpRet','EpCost','CostRate'])
        elif args.use_lag:
            file_name = f"{'lag'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='4_LAG',env_name=args.env,seed=args.seed,fieldnames=['EpRet','EpCost','CostRate'])
        elif args.use_fac:
            file_name = f"{'fac'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='5_FAC',env_name=args.env,seed=args.seed,fieldnames=['EpRet','EpCost','CostRate'])
        elif args.use_rs:
            file_name = f"{'rs'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='6_RS',env_name=args.env,seed=args.seed,fieldnames=['EpRet','EpCost','CostRate'])
        else:
            file_name = f"{'unconstrained'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='7_TD3',env_name=args.env,seed=args.seed,fieldnames=['EpRet','EpCost','CostRate'])
    else:
        file_name = args.exp_name
        logger = SafeLogger(exp_name=args.exp_name,env_name=args.env,seed=args.seed,fieldnames=['EpRet','EpCost','CostRate'])

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = create_env(args.seed)
    env._max_episode_steps = 250
    eval_env = deepcopy(env)
    
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "rew_discount": args.rew_discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
    }

    kwargs_safe = {
        "cost_discount": args.cost_discount,
        "delta": args.delta,                     
    }


    if args.use_usl:
        from saferl_algos.safetyplusplus import eval_policy
        kwargs.update(kwargs_safe)
        kwargs.update({'kappa':args.kappa})
        policy = saferl_algos.safetyplusplus.TD3Usl(**kwargs)
        replay_buffer = saferl_utils.CostReplayBuffer(state_dim, action_dim)
    elif args.use_recovery:
        from saferl_algos.recovery import eval_policy
        kwargs.update(kwargs_safe)
        policy = saferl_algos.recovery.TD3Recovery(**kwargs)
        replay_buffer = saferl_utils.RecReplayBuffer(state_dim, action_dim)
    elif args.use_qpsl:
        from saferl_algos.safetylayer import eval_policy
        kwargs.update(kwargs_safe)
        policy = saferl_algos.safetylayer.TD3Qpsl(**kwargs)
        replay_buffer = saferl_utils.SafetyLayerReplayBuffer(state_dim, action_dim)
    elif args.use_lag:
        from saferl_algos.lagrangian import eval_policy
        kwargs.update(kwargs_safe)
        policy = saferl_algos.lagrangian.TD3Lag(**kwargs)
        replay_buffer = saferl_utils.CostReplayBuffer(state_dim, action_dim)
    elif args.use_fac:
        from saferl_algos.fac import eval_policy
        kwargs.update(kwargs_safe)
        policy = saferl_algos.fac.TD3Fac(**kwargs)
        replay_buffer = saferl_utils.CostReplayBuffer(state_dim, action_dim)
    elif args.use_td3 or args.use_rs:
        from saferl_algos.unconstrained import eval_policy,TD3
        policy = TD3(**kwargs)
        replay_buffer = saferl_utils.SimpleReplayBuffer(state_dim, action_dim)
    else:
        raise NotImplementedError

    if args.load_model != "":
        policy.load(f"./models/{args.load_model}")
    
    reset_info, done = env.reset(), False
    state = reset_info[0]
    episode_reward = 0
    episode_cost = 0
    episode_timesteps = 0
    episode_num = 0
    cost_total = 0
    prev_cost = 0

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        if args.use_usl:
            if t < args.start_timesteps:
                action = env.action_space.sample()
            elif t < int(args.max_timesteps * args.warmup_ratio):
                action = policy.select_action(np.array(state),use_usl=False,exploration=True)
            else:
                action = policy.select_action(np.array(state),use_usl=True,exploration=True)
        elif args.use_recovery:
            if t < args.start_timesteps:
                raw_action = env.action_space.sample()
                action = raw_action
            elif t < int(args.max_timesteps * args.warmup_ratio):
                action,raw_action = policy.select_action(np.array(state),recovery=False,exploration=True)
            else:
                action,raw_action = policy.select_action(np.array(state),recovery=True,exploration=True)
        elif args.use_qpsl:
            if t < args.start_timesteps:
                action = env.action_space.sample()
            elif t < int(args.max_timesteps * args.warmup_ratio):
                action = policy.select_action(np.array(state),use_qpsl=False,exploration=True)
            else:
                action = policy.select_action(np.array(state),use_qpsl=True,prev_cost=prev_cost,exploration=True)
        else:
            if t < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(state),exploration=True)

    
        # Perform action
        next_state, reward, done, info = env.step(action)

        # cost value
        cost = 1. if info[args.flag] else 0.

        # if reward shaping
        if args.use_rs:
            reward -= args.cost_penalty * cost
        
        if cost > 0:
            cost_total += 1
            if args.early_stopping:
                done  = True

        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # set the early broken state as 'cost = 1'
        if done and episode_timesteps < env._max_episode_steps:
            cost = 1

        # Store data in replay buffer
        if args.use_td3 or args.use_rs:
            replay_buffer.add(state, action, next_state, reward, done_bool)
        elif args.use_recovery:
            replay_buffer.add(state, raw_action, action, next_state, reward, cost, done_bool)
        elif args.use_qpsl:
            replay_buffer.add(state, action, next_state, reward, cost, prev_cost, done_bool)
        else:
            replay_buffer.add(state, action, next_state, reward, cost, done_bool)
            

        state = next_state
        prev_cost = cost
        episode_reward += reward
        episode_cost += cost

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            if args.use_lag:
                print(f'Lambda : {policy.lam}')
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Cost: {episode_cost:.3f}")
            # Reset environment
            reset_info, done = env.reset(), False
            state = reset_info[0]
            episode_reward = 0
            episode_cost = 0
            episode_timesteps = 0
            episode_num += 1
            prev_cost = 0

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            if args.use_usl:
                evalEpRet,evalEpCost = eval_policy(policy, eval_env, args.seed, args.flag, use_usl=True)
            elif args.use_recovery:
                evalEpRet,evalEpCost = eval_policy(policy, eval_env, args.seed, args.flag, use_recovery=True)
            elif args.use_qpsl:
                evalEpRet,evalEpCost = eval_policy(policy, eval_env, args.seed, args.flag, use_qpsl=True)
            else:
                evalEpRet,evalEpCost = eval_policy(policy, eval_env, args.seed, args.flag)
            logger.update([evalEpRet,evalEpCost,1.0*cost_total/t], total_steps=t+1)
            if args.save_model:
                policy.save(f"./models/{file_name}")
