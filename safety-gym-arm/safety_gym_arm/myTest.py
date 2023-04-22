#!/usr/bin/env python

import argparse
import gym
import safety_gym_arm  # noqa
import numpy as np  # noqa
from safety_gym_arm.envs.engine import Engine
from getkey import getkey, keys

def run_random(env_name):
    # env = gym.make(env_name)
    config = {
        'robot_base': 'xmls/drone.xml',
        'goal_3D': True,
        # 'goal_travel': 1.5,
        # 'goal_mode': 'track',
        # 'num_steps': 2000,
        'arm_link_n': 3,
        'task': 'goal',
        'push_object': 'ball',
        # 'observe_goal_lidar': False,
        'compass_shape': 3,
        'goal_size': 0.5,
        'observe_goal_comp': True,
        # 'observe_box_lidar': False,
        # 'observe_box_comp': True,
        'observe_hazards': False,
        'observe_hazard3Ds': False,
        # 'observe_ghosts': True,
        'observe_ghost3Ds': False,
        'hazard3Ds_size': 0.3,
        'hazard3Ds_z_range': [0.5,1.5],
        'observe_vases': False,
        'constrain_hazards': False,
        'constrain_hazard3Ds': False,
        # 'observation_flatten': True,
        'lidar_max_dist': 4,
        'lidar_num_bins': 10,
        'lidar_num_bins3D': 6,
        # 'lidar_body': ['link_1', 'link_3', 'link_5','link_7'],
        'render_lidar_radius': 0.25,
        'hazard3Ds_num': 0,
        'hazard3Ds_size': 0.3,
        'hazard3Ds_keepout':0.5,
        # 'hazard3Ds_locations':[(0.0,1.5)],
        'hazards_num': 0,
        'hazards_keepout':0.5,
        'vases_num': 0,
        # 'vases_size': 0.2,
        # 'robot_locations':[(0.0,0.0)],
        'robot_rot':-1,
        'constrain_indicator':False,
        'hazards_cost':1.0,
        'gremlins_num': 0,


        'ghost3Ds_num': 0,
        'ghost3Ds_size': 0.2,
        'ghost3Ds_mode':'catch',
        'ghost3Ds_travel':2.5,
        'ghost3Ds_velocity': 0.0001,
        'ghost3Ds_z_range': [0.1, 3.1],
        'constrain_ghost3Ds': True,
        'ghost3Ds_contact':True,

        'constrain_ghosts': True,
        'ghosts_num': 0,
        'ghosts_size': 0.3,
        'ghosts_mode': 'catch',
        'ghosts_travel':1.5,
        'ghosts_velocity': 0.001,
        'ghosts_contact':False,
        
        'pillars_num': 0,
        'pillars_keepout': 0.3,
        'buttons_num': 0,
    }
    
    config1 = {
            'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
            'task': 'push',
            'push_object': 'ball',
            'observation_flatten': True,  # Flatten observation into a vector
            'observe_sensors': True,  # Observe all sensor data from simulator
            # Sensor observations
            # Specify which sensors to add to observation space
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
            'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
            'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
            'sensors_angle_components': True,  # Observe sin/cos theta instead of theta
            'continue_goal': False, # Done when goal is reached 
            
            'goal_size': 0.5,
            'observe_goal_comp': True,
            'observe_box_comp': True,
            'lidar_max_dist': 2,
            'lidar_num_bins': 16,
            'lidar_num_bins3D': 1,
            'render_lidar_radius': 0.25,
            'constrain_indicator':False,
            
            
            #num setting
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'hazards_num': 2,
            'hazards_size': 0.3,
            
            # Frameskip is the number of physics simulation steps per environment step
            # Frameskip is sampled as a binomial distribution
            # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
            'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) 
            'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
        }


        
    env = Engine(config)
    obs = env.reset()
    done = False
    ep_ret = 0
    ep_cost = 0
    cnt = 0
    T = 1000
    
    # with open("link_pos.txt",'w+') as f:
    #     pos_6 = []
    #     np.savetxt(f, pos_6,  delimiter = ' ')
    ac = 10.0
    while True:
        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f'%(ep_ret, ep_cost))
            ep_ret, ep_cost = 0, 0
            obs = env.reset()
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        # act = np.zeros(act.shape)
        
        # print(act)
        
        cnt = cnt + 1
        # if cnt % 1000 > 50:
        #     act = np.zeros(act.shape)
        # else:
        #     act = [0.0, -0.5, -0.5] 
        # if cnt % 400 > 200:
        #     act[0] = 10.0
        #     act[1] = 10.0
        # else:
        #     act[0] = -10.0

        #     act[1] = -10.0

        # if cnt != 0:
        #     key = getkey()
            # if key == "w":
            #     act[0] = 1.0
            # if key == "s":
            #     act[0] = -1.0
            # if key == "a":
            #     act[1] = 1.0
            # if key == "d":
            #     act[1] = -1.0
            # if key == "q":
            #     act[2] = 1.0
            # if key == "e":
            #     act[2] = -1.0  
        # cnt = 1
        # if cnt  > 100:
        #     act = [1.0, 1.0, -1.0, -1.0]  
        # if cnt > 200:
        #     act = [1.0, 1.0, -1.0, -1.0]
        # if cnt > 300:
        #     act = [1.0, 1.0, -1.0, -1.0]
        # if cnt > 400:
        #     act = [-1.0, -1.0, 1.0, 1.0]
        # if cnt > 500:
        #     act = [-1.0, -1.0, 1.0, 1.0]
        # if cnt > 600:
        #     act = [1.0, 1.0, 1.0, 1.0]
        # if cnt > 700:
        #     act = [0.0, 0.0, 0.0, 0.0]
        # if cnt > 800:
        #     act = [0.0, 0.0, 1.0, 1.0]
        # if cnt > 900:
        #     act = [0.0, 0.0, 0.0, 0.0]
        # cnt += 1
        # print(cnt, act)
        # act = [0.0, 0.0, 0.0]
        # if cnt  > 100:
        #     act = [0.5, 0.0, 0.0] 
        # if cnt > 800:
        #     act = [0.0, 0.0, 0.0]
        # if cnt%1000 > 300:
        #     act = [-0.1, -0.1, -0.2, -0.8, -0.8, 0.0]
        #     # act = [0.0, 0.0, 0.0, 0.0, -0.8, 0.0]
        # if cnt%1000 > 500:
        #     act = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # if cnt > 500:
        #     act = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0]
        # if cnt > 600:
        #     act = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # if cnt > 700:
        #     act = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        # if cnt > 800:
        #     act = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # if cnt > 900:
        #     act = [0.6, 0.2, -0.4, 0.0, 0.0, 0.6]

        # assert env.action_space.contains(act)
        obs, reward, done, info = env.step(act)
        # print(obs['accelerometer_link_2'])
        # joint = []
        # for i in range(6):
        #     joint.append(obs['jointpos_joint_'+str(i + 1)])
        # print(joint)
        # print('reward', reward)
        ep_ret += reward
        a = info['cost']
        # print(info.get('cost', 0))
        ep_cost += info.get('cost', 0)
        env.render()


if __name__ == '__main__':

    

    config = {
        'robot_base': 'xmls/point.xml',
        'task': 'push',
        'observe_goal_lidar': True,
        'observe_box_lidar': True,
        'observe_hazards': True,
        'observe_vases': True,
        'constrain_hazards': True,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'hazards_num': 4,
        'vases_num': 4
    }

    env = Engine(config)
    run_random(env)

    # from gym.envs.registration import register

    # register(id='SafexpTestEnvironment-v0',
    #         entry_point='safety_gym_arm.envs.mujoco:Engine',
    #         kwargs={'config': config})
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env', default='SafexpTestEnvironment-v0')
    # args = parser.parse_args()
    # run_random(args.env)
