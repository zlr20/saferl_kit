#!/usr/bin/env python

import argparse
import gym
import safety_gym_arm  # noqa
import numpy as np  # noqa
from safety_gym_arm.envs.engine import Engine
from getkey import getkey, keys
from my_config import configuration

def run_random(env_name):
    # env = gym.make(env_name)
    # config = {
    #     'robot_base': 'xmls/arm_6.xml',
    #     # 'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
    #     'goal_3D': True,
    #     # 'goal_travel': 1.5,
    #     # 'goal_mode': 'track',
    #     # 'num_steps': 2000,
    #     'task': 'push',
    #     'push_object': 'ball',
    #     'observe_goal_lidar': True,
    #     'compass_shape': 3,
    #     'goal_size': 0.5,
    #     'observe_goal_comp': True,
    #     # 'observe_box_lidar': False,
    #     # 'observe_box_comp': True,
    #     'observe_hazards': False,
    #     'observe_pillars': False,
    #     'observe_hazard3Ds': True,
    #     # 'observe_ghosts': True,
    #     'observe_ghost3Ds': True,
    #     'hazard3Ds_size': 0.3,
    #     'hazard3Ds_z_range': [0.5,1.5],
    #     'observe_vases': False,
    #     'constrain_hazards': False,
    #     'constrain_hazard3Ds': True,
    #     'observation_flatten': False,
    #     'lidar_max_dist': 4,
    #     'lidar_num_bins': 10,
    #     'lidar_num_bins3D': 6,
    #     'render_lidar_radius': 0.25,
    #     'hazard3Ds_num': 0,
    #     'hazard3Ds_size': 0.3,
    #     'hazard3Ds_keepout':0.5,
    #     # 'hazard3Ds_locations':[(0.0,1.5)],
    #     'hazards_num': 0,
    #     # 'hazards_keepout':0.5,
    #     'vases_num': 0,
    #     # 'vases_size': 0.2,
    #     # 'robot_locations':[(0.0,0.0)],
    #     'robot_rot':-1,
    #     'constrain_indicator':False,
    #     'hazards_cost':1.0,
    #     'gremlins_num': 0,


    #     'ghost3Ds_num': 8,
    #     'ghost3Ds_size': 0.2,
    #     # 'ghost3Ds_mode':'catch',
    #     'ghost3Ds_keepout': 0.2,
    #     'ghost3Ds_travel':3.0,
    #     'ghost3Ds_velocity': 0.001,
    #     'ghost3Ds_z_range': [0.1, 3.0],
    #     'constrain_ghost3Ds': True,
    #     'ghost3Ds_contact':False,

    #     'constrain_ghosts': True,
    #     'ghosts_num': 0,
    #     'ghosts_size': 0.3,
    #     # 'ghosts_mode': 'catch',
    #     'ghosts_travel':1.5,
    #     'ghosts_velocity': 0.001,
    #     'ghosts_contact':False,

    #     # 'constrain_robbers': True,
    #     'robbers_num': 2,
    #     'robbers_size': 0.3,
    #     # 'robbers_mode': 'catch',
    #     'robbers_travel':0,
    #     'robbers_velocity': 0.001,
    #     'robbers_contact':False,

    #     'robber3Ds_num': 0,
    #     'robber3Ds_size': 0.3,
    #     # 'robbers_mode': 'catch',
    #     'robber3Ds_travel':1.5,
    #     'robber3Ds_velocity': 0.001,
    #     'robber3Ds_contact':False,
    #     'robber3Ds_z_range': [0.1, 1.1],
        
    #     'pillars_num': 0,
    #     'pillars_keepout': 0.3,
    #     'buttons_num': 0,
    # }
    
    config = {
            # robot setting
            'robot_base': 'xmls/swimmer.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            # 'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
            #                 'touch_point1', 'touch_point2', 'touch_point3', 'touch_point4'],

            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_vases': True,
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            # 'hazards_num': 8,
            'hazards_size': 0.3,

            'vases_num': 8,
        }


    goal_task_list = ['Goal_Point_8Hazards',
                 'Goal_Point_8Ghosts',
                 'Goal_Swimmer_8Hazards',
                 'Goal_Swimmer_8Ghosts',
                 'Goal_Ant_8Hazards',
                 'Goal_Ant_8Ghosts',
                 'Goal_Walker_8Hazards',
                 'Goal_Walker_8Ghosts',
                 'Goal_Humanoid_8Hazards',
                 'Goal_Humanoid_8Ghosts',
                 'Goal_Hopper_8Hazards',
                 'Goal_Hopper_8Ghosts',
                 'Goal_Arm3_8Hazards',
                 'Goal_Arm3_8Ghosts',
                 'Goal_Arm6_8Hazards',
                 'Goal_Arm6_8Ghosts',
                 'Goal_Drone_8Hazards',
                 'Goal_Drone_8Ghosts']
    
    push_task_list = ['Push_Point_8Hazards',
                 'Push_Point_8Ghosts',
                 'Push_Swimmer_8Hazards',
                 'Push_Swimmer_8Ghosts',
                 'Push_Ant_8Hazards',
                 'Push_Ant_8Ghosts',
                 'Push_Walker_8Hazards',
                 'Push_Walker_8Ghosts',
                 'Push_Humanoid_8Hazards',
                 'Push_Humanoid_8Ghosts',
                 'Push_Hopper_8Hazards',
                 'Push_Hopper_8Ghosts',
                 'Push_Arm3_8Hazards',
                 'Push_Arm3_8Ghosts',
                 'Push_Arm6_8Hazards',
                 'Push_Arm6_8Ghosts',
                 'Push_Drone_8Hazards',
                 'Push_Drone_8Ghosts']
    
    chase_task_list = ['Chase_Point_8Hazards',
                 'Chase_Point_8Ghosts',
                 'Chase_Swimmer_8Hazards',
                 'Chase_Swimmer_8Ghosts',
                 'Chase_Ant_8Hazards',
                 'Chase_Ant_8Ghosts',
                 'Chase_Walker_8Hazards',
                 'Chase_Walker_8Ghosts',
                 'Chase_Humanoid_8Hazards',
                 'Chase_Humanoid_8Ghosts',
                 'Chase_Hopper_8Hazards',
                 'Chase_Hopper_8Ghosts',
                 'Chase_Arm3_8Hazards',
                 'Chase_Arm3_8Ghosts',
                 'Chase_Arm6_8Hazards',
                 'Chase_Arm6_8Ghosts',
                 'Chase_Drone_8Hazards',
                 'Chase_Drone_8Ghosts']
    
    defense_task_list = ['Defense_Point_8Hazards',
                 'Defense_Point_8Ghosts',
                 'Defense_Swimmer_8Hazards',
                 'Defense_Swimmer_8Ghosts',
                 'Defense_Ant_8Hazards',
                 'Defense_Ant_8Ghosts',
                 'Defense_Walker_8Hazards',
                 'Defense_Walker_8Ghosts',
                 'Defense_Humanoid_8Hazards',
                 'Defense_Humanoid_8Ghosts',
                 'Defense_Hopper_8Hazards',
                 'Defense_Hopper_8Ghosts',
                 'Defense_Arm3_8Hazards',
                 'Defense_Arm3_8Ghosts',
                 'Defense_Arm6_8Hazards',
                 'Defense_Arm6_8Ghosts',
                 'Defense_Drone_8Hazards',
                 'Defense_Drone_8Ghosts']

    config = configuration("Chase_Hopper_8Hazards")
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
        # print(obs['touch_point1'])
        # print(obs['accelerometer_link_2'])
        # joint = []
        # for i in range(6):
        #     joint.append(obs['jointpos_joint_'+str(i + 1)])
        # print(joint)
        # print('reward', reward)
        print(len(obs))
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
