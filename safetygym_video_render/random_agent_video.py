#!/usr/bin/env python
import os 
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
# os.environ['DISPLAY'] = ''
import argparse
import gym
import safety_gym  # noqa
from safety_gym.envs.engine import Engine
import numpy as np  # noqa
import cv2
import matplotlib.pyplot as plt
from mujoco_py import GlfwContext


def random_agent_save_video(args):
    # create a video folder 
    env_name = args.env
    
    path = "video"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    # random agent 
    if not args.task:
        env = gym.make(env_name)
    else:
        env = Engine(configuration(args.task, args))
        
    obs = env.reset()
    done = False
    ep_ret = 0
    ep_cost = 0
    time_step = 0

    video_array = []

    while True:
        time_step += 1
        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f'%(ep_ret, ep_cost))
            env.close()
            break

            ep_ret, ep_cost = 0, 0
            obs = env.reset()
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        obs, reward, done, info = env.step(act)

        ################ save video without display ################
        # img_array = env.render(mode='rgb_array', width=1920, height=1200)
        img_array = env.render(mode='rgb_array')
        video_array.append(img_array)

        ep_ret += reward
        ep_cost += info.get('cost', 0)

    # save video 
    fps = 60
    dsize = (1920,1080)
    out_path = '/home/safebench/safety_benchmark/safety-gym/safety_gym/output/video'
    # video_writer = cv2.VideoWriter(os.path.join(out_path, args.video_name, '.mp4'),
    #                             cv2.VideoWriter_fourcc(*'FMP4'), fps, dsize)
    video_writer = cv2.VideoWriter(os.path.join(out_path,'test_video_mygoal.mp4'),
                                cv2.VideoWriter_fourcc(*'FMP4'), fps, dsize)

    for frame in video_array:
        resized = cv2.resize(frame, dsize=dsize)
        video_writer.write(resized)

    video_writer.release()


def run_random(args):
    # random agent 
    
    env_name = args.env
    
    # make env
    # random agent 
    if not args.task:
        env = gym.make(env_name)
    else:
        env = Engine(configuration(args.task, args))
        
    obs = env.reset()
    done = False
    ep_ret = 0
    ep_cost = 0
    ep_cost_avg = 0
    time_step = 0
    while True:
        time_step += 1
        if done:
            print('episode ends')
            print('Episode Return: %.3f \t Episode Cost: %.3f'%(ep_ret, ep_cost))
            print(f"episode cost average: {ep_cost_avg}")
            ep_ret, ep_cost = 0, 0
            ep_cost_avg = 0
            obs = env.reset()
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        obs, reward, done, info = env.step(act)

        ep_ret += reward
        ep_cost += info.get('cost', 0)
        ep_cost_avg += info['cost']
        
        # make sure the equalty 
        assert info.get('cost', 0) == info['cost']
        # if info['cost'] > 0.:
        #     print(f"collision happens, cost is {info['cost']}")
        
        
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Safexp-PointGoal1-v0')
    parser.add_argument('--random', action="store_true")
    parser.add_argument('--video', action="store_true")
    parser.add_argument('--task', default='Mygoal1')  
    parser.add_argument('--hazards_size', type=float, default=0.30)
    parser.add_argument('--video_name', default='test_video')
    args = parser.parse_args()
    
    if args.random:
        run_random(args)
        
    if args.video:
        random_agent_save_video(args)
