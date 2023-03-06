import os,sys,argparse,warnings
warnings.filterwarnings("ignore")
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import numpy as np
import gym
import torch
import saferl_algos
from saferl_plotter.logger import SafeLogger, RunTimeLogger
import saferl_utils
from saferl_plotter import log_utils as lu
import safety_gym
from safety_gym.envs.engine import Engine



######################### make cartpole stabilization environment ################################################################
sys.path.append("saferl_envs")
from copy import deepcopy
from functools import partial

##################################################################################################################################


"""
configuration for customized environment for safety gym 
"""
def configuration(task, args):
    if task == "Mygoal1":
        config = {
            'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
            'task': 'goal',
            'observation_flatten': True,  # Flatten observation into a vector
            'observe_sensors': True,  # Observe all sensor data from simulator
            # Sensor observations
            # Specify which sensors to add to observation space
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
            'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
            'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
            'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

            #observe goal/box/...
            'observe_goal_dist': False,  # Observe the distance to the goal
            'observe_goal_comp': False,  # Observe a compass vector to the goal
            'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': False,  # Observe the box with a compass
            'observe_box_lidar': False,  # Observe the box with a lidar
            'observe_circle': False,  # Observe the origin with a lidar
            'observe_remaining': False,  # Observe the fraction of steps remaining
            'observe_walls': False,  # Observe the walls with a lidar space
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'observe_vases': True,  # Observe the vector from agent to vases
            'observe_pillars': False,  # Lidar observation of pillar object positions
            'observe_buttons': False,  # Lidar observation of button object positions
            'observe_gremlins': False,  # Gremlins are observed with lidar-like space
            'observe_vision': False,  # Observe vision from the robot

            # Constraints - flags which can be turned on
            # By default, no constraints are enabled, and all costs are indicator functions.
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_vases': False,  # Constrain frobot from touching objects
            'constrain_pillars': False,  # Immovable obstacles in the environment
            'constrain_buttons': False,  # Penalize pressing incorrect buttons
            'constrain_gremlins': False,  # Moving objects that must be avoided
            # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            #lidar setting
            'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
            'lidar_num_bins': 16,
            #num setting
            'hazards_num': 1,
            'hazards_size': args.hazards_size,
            'vases_num': 0,



            # Frameskip is the number of physics simulation steps per environment step
            # Frameskip is sampled as a binomial distribution
            # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
            'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) 
            'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
        }

    if task == "Mygoal4":
        config = {
            'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
            'task': 'goal',
            'observation_flatten': True,  # Flatten observation into a vector
            'observe_sensors': True,  # Observe all sensor data from simulator
            # Sensor observations
            # Specify which sensors to add to observation space
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
            'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
            'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
            'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

            #observe goal/box/...
            'observe_goal_dist': False,  # Observe the distance to the goal
            'observe_goal_comp': False,  # Observe a compass vector to the goal
            'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': False,  # Observe the box with a compass
            'observe_box_lidar': False,  # Observe the box with a lidar
            'observe_circle': False,  # Observe the origin with a lidar
            'observe_remaining': False,  # Observe the fraction of steps remaining
            'observe_walls': False,  # Observe the walls with a lidar space
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'observe_vases': True,  # Observe the vector from agent to vases
            'observe_pillars': False,  # Lidar observation of pillar object positions
            'observe_buttons': False,  # Lidar observation of button object positions
            'observe_gremlins': False,  # Gremlins are observed with lidar-like space
            'observe_vision': False,  # Observe vision from the robot

            # Constraints - flags which can be turned on
            # By default, no constraints are enabled, and all costs are indicator functions.
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_vases': False,  # Constrain frobot from touching objects
            'constrain_pillars': False,  # Immovable obstacles in the environment
            'constrain_buttons': False,  # Penalize pressing incorrect buttons
            'constrain_gremlins': False,  # Moving objects that must be avoided
            # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            #lidar setting
            'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
            'lidar_num_bins': 16,
            #num setting
            'hazards_num': 4,
            'hazards_size': args.hazards_size,
            'vases_num': 0,



            # Frameskip is the number of physics simulation steps per environment step
            # Frameskip is sampled as a binomial distribution
            # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
            'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) 
            'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
        }

    return config

def create_env(args):
    if not args.task:
        env = gym.make(args.safety_default)
    else:
        env = Engine(configuration(args.task, args))
    return env


# python main.py --task cartpole --overrides saferl_envs/config_overrides/stable_task.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--safety_default", type=str, default="Safexp-PointGoal1-v0")
    parser.add_argument("--task", type=str, default="Mygoal1")
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
    parser.add_argument("--use_cpo",action="store_true")            # Wether to use Constrained Policy Optimization
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    # Hyper-parameters for all safety-aware algorithms
    parser.add_argument("--delta",default = 1,type=float)         # Qc(s,a) \leq \delta
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
    parser.add_argument('--hazards_size', type=float, default=0.30)  # the default hazard size of safety gym 
    # Hyper-parameters for using CPO
    parser.add_argument("--backtrack_coeff",default = 0.8)          # Backtrack coefficient
    parser.add_argument("--kl_div_lim",default = 1e-3, type=float)              # the limit of KL divergence
    parser.add_argument("--backtrack_iters",default = 10)           # Backtrack iterations for CPO line search update   
    parser.add_argument("--delay", default = 5, type=int)                      # delayed update for the random policy    
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
    parser.add_argument("--display_timestep_freq", default=500, type = int)   # Frequency of time step display in during training
    args = parser.parse_args()

    assert [bool(i) for i in [args.use_td3,
                              args.use_usl,
                              args.use_recovery,
                              args.use_qpsl,
                              args.use_lag,
                              args.use_fac,
                              args.use_rs,
                              args.use_cpo]].count(True) == 1, 'Only one option can be True'


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
        elif args.use_cpo:
            file_name = f"{'cpo'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='7_CPO',env_name=f"{args.env}_delay{args.delay}_kllim{args.kl_div_lim}_costlim{args.delta}",seed=args.seed,fieldnames=['EpRet','EpCost','CostRate'])
            # logger = SafeLogger(exp_name='7_CPO',env_name=f"{args.env}_delay{args.delay}",seed=args.seed,fieldnames=['EpRet','EpCost','CostRate'])
            runtime_logger = RunTimeLogger() # log the EpCost for each episode
            runtime_transition_logger = RunTimeLogger() # log the transition for each episode
        else:
            file_name = f"{'unconstrained'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='8_TD3',env_name=args.env,seed=args.seed,fieldnames=['EpRet','EpCost','CostRate'])
    else:
        file_name = args.exp_name
        logger = SafeLogger(exp_name=args.exp_name,env_name=args.env,seed=args.seed,fieldnames=['EpRet','EpCost','CostRate'])

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    '''
    create environment
    '''
    env = create_env(args)
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
    
    kwargs_cpo = {
        "cost_lim": args.delta,
        "cost_discount": args.cost_discount,
        "rew_discount": args.rew_discount,
        "backtrack_coeff": args.backtrack_coeff,
        "delta": args.kl_div_lim,
        "backtrack_iters": args.backtrack_iters
    }
    
    '''
    set algorithm
    '''


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
    elif args.use_cpo:
        from saferl_algos.cpo import eval_policy
        kwargs.update(kwargs_cpo)
        policy = saferl_algos.cpo.CPO(**kwargs)
        replay_buffer = saferl_utils.CPOReplayBuffer(state_dim, action_dim)
    else:
        raise NotImplementedError

    if args.load_model != "":
        policy.load(f"./models/{args.load_model}")
    
    if args.flag == 'safety_gym':
        state, done = env.reset(), False
    else:
        reset_info, done = env.reset(), False
        state = reset_info[0]
    episode_reward = 0
    episode_cost = 0
    episode_discount_cost = 0
    episode_timesteps = 0
    episode_num = 0
    cost_total = 0
    prev_cost = 0

    for t in range(int(args.max_timesteps)):
        
        if episode_timesteps % args.display_timestep_freq == 0:
            print(lu.colorize(f"current time step: {t}", 'yellow', bold=True))

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
        # cost = 1. if info[args.flag] else 0.
        cost = info['cost']

        # if reward shaping
        if args.use_rs:
            reward -= args.cost_penalty * cost
        
        if cost > 0:
            cost_total += cost
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
        elif args.use_cpo:
            # instead of CPO related runtime logger, vanilla PG just care about the reward
            # instead of add replay_buffer, CPO maintains the run time logger and save buffer at the end of each episode
            runtime_transition_logger.update(**{"state":state, "action":action, "next_state":next_state, "reward":reward, "cost":cost, "done_bool":done_bool})
        else:
            replay_buffer.add(state, action, next_state, reward, cost, done_bool)
            

        state = next_state
        prev_cost = cost
        episode_reward += reward
        episode_cost += cost
        episode_discount_cost += cost * args.cost_discount**(episode_timesteps-1)

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            if args.use_cpo:
                pass
            else:  
                policy.train(replay_buffer, args.batch_size)

        if done: 
            if args.use_cpo:
                # Save the episode cost at each end of the episode
                # ! Here the episode cost is not discounted sum, but direct sum
                # runtime_logger.update(**{"EpCost": episode_cost, "EpLen": episode_timesteps})
                runtime_logger.update(**{"EpCost": episode_discount_cost, "EpLen": episode_timesteps})
                
                # update the replay buffer with the episode transition from logger 
                # ! Currently cost to go and reward to go has the discount factor of 1
                # update the cost_to_go from rum time logger 
                runtime_transition_logger.update_value_to_go("cost", discount=args.cost_discount)
                # udpate the reward_to_go from rum time logger 
                runtime_transition_logger.update_value_to_go("reward", discount=args.rew_discount)
                # load the episode information to replay buffer 
                episode_transition_logger = runtime_transition_logger.get_complete_stats()
                for i in range(runtime_transition_logger.len):
                    replay_buffer.add(episode_transition_logger["state"][i],
                                      episode_transition_logger["action"][i],
                                      episode_transition_logger["next_state"][i],
                                      episode_transition_logger["reward"][i],
                                      episode_transition_logger["cost"][i],
                                      episode_transition_logger["cost_to_go"][i],
                                      episode_transition_logger["reward_to_go"][i],
                                      episode_transition_logger["done_bool"][i])
                # refresh the runtime trnasition logger at end of each episode
                runtime_transition_logger.reset()
                assert runtime_transition_logger.empty == True
                
                # update the policy for multiple rounds training 
                if not runtime_logger.empty:
                    # ! first update the policy 
                    for _ in range(max(1, episode_timesteps // args.delay)):
                        # train enough actor
                        policy.train_cpo_policy(replay_buffer, args.batch_size, runtime_logger)
                    # ! next update the value function
                    for _ in range(episode_timesteps):
                        # train enough critic 
                        policy.train_cpo_critic(replay_buffer, args.batch_size)
                    
                
                    
            if args.use_lag:
                print(f'Lambda : {policy.lam}')
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print("---------------------------------------")
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Cost: {episode_cost:.3f}")
            print("---------------------------------------")
            # Reset environment
            if args.flag == 'safety_gym':
                state, done = env.reset(), False
            else:
                reset_info, done = env.reset(), False
                state = reset_info[0]
            episode_reward = 0
            episode_cost = 0
            episode_discount_cost = 0
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
