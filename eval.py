import os,argparse
import numpy as np
import gym
import torch
import bullet_safety_gym
import utils
import TD3
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filt
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="SafetyCarAvoidance-v0")          # OpenAI gym environment name
    parser.add_argument("--seed", default=777, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--render",action="store_true") 
    parser.add_argument("--load_model", default="lag_TD3_SafetyCarAvoidance-v0_0")             # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--use_usl",action="store_true")
    parser.add_argument("--eta", default = 0.05)                     # Gradient step factor for USL
    parser.add_argument("--kappa",default = 10)                     # Penalized factor for soft loss function
    parser.add_argument("--delta",default = 0.1)                    # Qc(s,a) \leq \delta
    parser.add_argument("--K_test", default = 20)
    args = parser.parse_args()

    env = gym.make(args.env)
    env.seed(args.seed)
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
        # "eta": args.eta,
        # "kappa": args.kappa,
        # "delta": args.delta,      
    }
    
    # Initialize policy
    
    policy_file = args.load_model
    policy = TD3.TD3(**kwargs)
    policy.load(f"./models/{policy_file}")
    
    # Evaluate pretrained policy
    for iter in range(10):
        env.render()
        state, done = env.reset(), False
        ep_reward = 0
        ep_cost = 0

        t = 0
        while not done:
            t += 1
            action = policy.select_action(np.array(state))
            if args.use_usl:
                action = policy.select_action(np.array(state),use_usl=False)
            #pred_c = policy.pred_cost(np.array(state),np.array(action))
            next_state, reward, done, info = env.step(action)
            gt_c = info['cost']
            if args.render:
                env.render()
                time.sleep(1/60)
            state = next_state
            ep_reward += reward
            ep_cost += info['cost']

            
        print(f'Episode Reward : {ep_reward}, Episode Cost : {ep_cost}')