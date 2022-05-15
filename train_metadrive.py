import os, argparse, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
warnings.filterwarnings("ignore")
import numpy as np
import gym
import torch
import saferl_algos
from saferl_plotter.logger import SafeLogger
import saferl_utils
from metadrive import SafeMetaDriveEnv


def eval_policy(policy, policy_type, eval_env, seed, eval_episodes=20):
    avg_reward = 0.
    avg_cost = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        t = 0
        while not (done or t >= eval_env._max_episode_steps):
            if policy_type == 'use_usl':
                action = policy.select_action(np.array(state), use_usl=True)
            elif policy_type == 'use_qpsl':
                action = policy.select_action(np.array(state), use_qpsl=True)
            elif policy_type == 'use_recovery':
                action, raw_action = policy.select_action(np.array(state), recovery=True)
            else:  # default td3
                action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            if info['cost'] != 0:
                avg_cost += info['cost']
            t = t + 1
        # success rate
        if info['arrive_dest']:
            avg_reward += 1

    avg_reward /= eval_episodes
    avg_cost /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} Cost {avg_cost:.3f}.\
             SafeRL used : {policy_type}")
    print("---------------------------------------")
    return avg_reward, avg_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--env", default="SafeMetaDriveEnv")  # Env name
    parser.add_argument("--seed", default=1000, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--device", default='3')  # cuda device
    parser.add_argument("--base_policy", default="TD3")  # Base Policy name
    parser.add_argument("--use_td3", action="store_true")  # unconstrained RL
    parser.add_argument("--use_epo", action="store_true")  # Wether to use Exact Penalized Function
    parser.add_argument("--use_usl", action="store_true")  # Wether to use Unrolling Safety Layer (TODO)
    parser.add_argument("--use_qpsl", action="store_true")  # Wether to use QP Safety Layer (Dalal 2018)
    parser.add_argument("--use_recovery", action="store_true")  # Wether to use Recovery RL     (Thananjeyan 2021)
    parser.add_argument("--use_lag", action="store_true")  # Wether to use Lagrangian Relaxation  (Ray 2019)
    parser.add_argument("--use_fac", action="store_true")  # Wether to use FAC (Ma 2021)
    # Hyper-parameters for all safety-aware algorithms
    parser.add_argument("--delta", default=0.1, type=float)  # Qc(s,a) \leq \delta
    parser.add_argument("--cost_discount", default=0.99)  # Discount factor for cost-return
    # Hyper-parameters for using epo
    parser.add_argument("--warmup_ratio",
                        default=1 / 5)  # Start using USL in traing after max_timesteps*warmup_ratio steps
    # Other hyper-parameters for original TD3
    parser.add_argument("--start_timesteps", default=10000, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=10000, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--rew_discount", default=0.99)  # Discount factor for reward-return
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name

    args = parser.parse_args()

    assert [bool(i) for i in [args.use_td3, args.use_epo, args.use_usl, args.use_recovery, args.use_qpsl, args.use_lag,
                              args.use_fac]].count(True) == 1, 'Only one option can be True'



    if not args.exp_name:
        if args.use_usl:
            file_name = f"{'usl'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='0_USL', env_name=args.env, seed=args.seed,
                                fieldnames=['EpRet', 'EpCost', 'CostRate'])
        elif args.use_epo:
            file_name = f"{'epo'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='1_epo', env_name=args.env, seed=args.seed,
                                fieldnames=['EpRet', 'EpCost', 'CostRate'])
        elif args.use_recovery:
            file_name = f"{'rec'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='2_REC', env_name=args.env, seed=args.seed,
                                fieldnames=['EpRet', 'EpCost', 'CostRate'])
        elif args.use_qpsl:
            file_name = f"{'qpsl'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='3_QPSL', env_name=args.env, seed=args.seed,
                                fieldnames=['EpRet', 'EpCost', 'CostRate'])
        elif args.use_lag:
            file_name = f"{'lag'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='4_LAG', env_name=args.env, seed=args.seed,
                                fieldnames=['EpRet', 'EpCost', 'CostRate'])
        elif args.use_fac:
            file_name = f"{'fac'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='5_FAC', env_name=args.env, seed=args.seed,
                                fieldnames=['EpRet', 'EpCost', 'CostRate'])
        else:
            file_name = f"{'td3'}_{args.base_policy}_{args.env}_{args.seed}"
            logger = SafeLogger(exp_name='6_TD3', env_name=args.env, seed=args.seed,
                                fieldnames=['EpRet', 'EpCost', 'CostRate'])
    else:
        file_name = args.exp_name
        logger = SafeLogger(exp_name=args.exp_name, env_name=args.env, seed=args.seed,
                            fieldnames=['EpRet', 'EpCost', 'CostRate'])

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    config_train = dict(
        environment_num=20,
        start_seed=args.seed,
        cost_to_reward=True,
        traffic_density=0.12,
        vehicle_config=dict(
            lidar=dict(num_lasers=30)),
    )
    config_test = dict(
        environment_num=20,
        start_seed=args.seed + 100,
        cost_to_reward=True,
        traffic_density=0.12,
        vehicle_config=dict(
            lidar=dict(num_lasers=30)),
    )

    env = SafeMetaDriveEnv(config=config_train)
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

    run_policy_type = ''
    if args.use_usl:
        run_policy_type = 'use_usl'
        # from saferl_algos.unrolling import eval_policy
        kwargs.update(kwargs_safe)
        policy = saferl_algos.unrolling.TD3Usl(**kwargs)
        replay_buffer = saferl_utils.CostReplayBuffer(state_dim, action_dim)
    elif args.use_epo:
        run_policy_type = 'use_epo'
        kwargs.update(kwargs_safe)
        policy = saferl_algos.exactpenalty.TD3epo(**kwargs)
        replay_buffer = saferl_utils.CostReplayBuffer(state_dim, action_dim)
    elif args.use_recovery:
        run_policy_type = 'use_recovery'
        kwargs.update(kwargs_safe)
        policy = saferl_algos.recovery.TD3Recovery(**kwargs)
        replay_buffer = saferl_utils.RecReplayBuffer(state_dim, action_dim)
    elif args.use_qpsl:
        run_policy_type = 'use_qpsl'
        kwargs.update(kwargs_safe)
        policy = saferl_algos.safetylayer.TD3Qpsl(**kwargs)
        replay_buffer = saferl_utils.SafetyLayerReplayBuffer(state_dim, action_dim)
    elif args.use_lag:
        run_policy_type = 'use_lag'
        kwargs.update(kwargs_safe)
        policy = saferl_algos.lagrangian.TD3Lag(**kwargs)
        replay_buffer = saferl_utils.CostReplayBuffer(state_dim, action_dim)
    elif args.use_fac:
        run_policy_type = 'use_fac'
        kwargs.update(kwargs_safe)
        policy = saferl_algos.fac.TD3Fac(**kwargs)
        replay_buffer = saferl_utils.CostReplayBuffer(state_dim, action_dim)
    elif args.use_td3:
        run_policy_type = 'use_td3'
        policy = saferl_algos.unconstrained.TD3(**kwargs)
        replay_buffer = saferl_utils.SimpleReplayBuffer(state_dim, action_dim)
    else:
        raise NotImplementedError

    if args.load_model != "":
        policy.load(f"./models/{args.load_model}")

    state, done = env.reset(), False
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
                action = policy.select_action(np.array(state), use_usl=False, exploration=True)
            else:
                action = policy.select_action(np.array(state), use_usl=True, exploration=True)
        elif args.use_recovery:
            if t < args.start_timesteps:
                raw_action = env.action_space.sample()
                action = raw_action
            elif t < int(args.max_timesteps * args.warmup_ratio):
                action, raw_action = policy.select_action(np.array(state), recovery=False, exploration=True)
            else:
                action, raw_action = policy.select_action(np.array(state), recovery=True, exploration=True)
        elif args.use_qpsl:
            if t < args.start_timesteps:
                action = env.action_space.sample()
            elif t < int(args.max_timesteps * args.warmup_ratio):
                action = policy.select_action(np.array(state), use_qpsl=False, exploration=True)
            else:
                action = policy.select_action(np.array(state), use_qpsl=True, prev_cost=prev_cost, exploration=True)
        else:
            if t < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(state), exploration=True)

        # Perform action
        next_state, reward, done, info = env.step(action)

        # cost value
        cost = info['cost']

        if cost > 0:
            cost_total += 1

        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        if args.use_td3:
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

        if done or episode_timesteps >= env._max_episode_steps:
            print(f"arrive destination: {info['arrive_dest']} ,out of road:{info['out_of_road']}  ", end=' ')
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Cost: {episode_cost:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_cost = 0
            episode_timesteps = 0
            episode_num += 1
            prev_cost = 0

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            env.close()
            eval_env = SafeMetaDriveEnv(config=config_test)
            evalEpRet, evalEpCost = eval_policy(policy, run_policy_type, eval_env, args.seed)
            eval_env.close()
            env = SafeMetaDriveEnv(config=config_train)
            state, done = env.reset(), False
            logger.update([evalEpRet, evalEpCost, 1.0 * cost_total / t], total_steps=t + 1)
            if args.save_model:
                policy.save(f"./models/{file_name}")
