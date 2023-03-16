import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import ppo_core as core
from utils.logx import EpochLogger, setup_logger_kwargs
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from safety_gym.envs.engine import Engine as safety_gym_Engine
from safety_gym_arm.envs.engine import Engine as safety_gym_arm_Engine
from utils.safetygym_config import configuration
import os.path as osp
import cv2
import matplotlib.pyplot as plt
from mujoco_py import GlfwContext

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def create_env(args):
    if 'Arm' in args.task:
        env = safety_gym_arm_Engine(configuration(args.task, args))
    else:
        env = safety_gym_Engine(configuration(args.task, args))
    return env


def replay(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), model_path=None, video_name=None):
    if not model_path:
        print("please specify a model path")
        raise NotImplementedError
    if not video_name:
        print("please specify a video name")
        raise NotImplementedError    
    
    # Instantiate environment
    env = env_fn()
    
    # reset environment
    o = env.reset()
    d = False
    ep_ret = 0
    time_step = 0
    
    video_array = []
    
    # load the model 
    ac = torch.load(model_path)
    
    # evaluate the model 
    while True:
        time_step += 1
        if d:
            print('Episode Return: %.3f'%(ep_ret))
            env.close()
            break

            ep_ret = 0
            o = env.reset()
        
        try:
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
        except:
            print('please choose the correct environment, the observation space doesn''t match')
            raise NotImplementedError
        

        next_o, r, d, _ = env.step(a)
        
        # Update obs (critical!)
        o = next_o

        img_array = env.render(mode='rgb_array')
        video_array.append(img_array)

        ep_ret += r

    # save video 
    fps = 60
    dsize = (1920,1080)
    out_path = '../video'
    existence = os.path.exists(out_path)
    if not existence:
        os.makedirs(out_path)
    video_writer = cv2.VideoWriter(os.path.join(out_path,f'{video_name}.mp4'),
                                cv2.VideoWriter_fourcc(*'FMP4'), fps, dsize)

    for frame in video_array:
        resized = cv2.resize(frame, dsize=dsize)
        video_writer.write(resized)

    video_writer.release()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task', type=str, default='Mygoal4')
    parser.add_argument('--hazards_size', type=float, default=0.30)  # the default hazard size of safety gym 
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--video_name', type=str, default=None)
    args = parser.parse_args()

    replay(lambda : create_env(args), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), model_path=args.model_path, video_name=args.video_name)