#!/usr/bin/env python

import argparse
import gym
import safety_gym_arm  # noqa
import numpy as np  # noqa
from safety_gym_arm.envs.engine import Engine


def run_random(env_name):
    # env = gym.make(env_name)
    config = {
        'robot_base': 'xmls/arm_3.xml',
        'arm_link_n': 5,
        'task': 'goal',
        'goal_3D': True,
        'goal_locations': [(0.0,-1.0)],
        'observe_goal_lidar': False,
        'compass_shape': 3,
        'goal_size': 0.5,
        'observe_goal_comp': True,
        # 'observe_box_lidar': False,
        # 'observe_box_comp': True,
        'observe_hazard3Ds': False,
        'observe_vases': False,
        'constrain_hazards': False,
        'constrain_hazard3Ds': True,
        'observation_flatten': False,
        # 'continue_goal': True,
        # 'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
        #                 'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
        #                 'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
        #                 'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
        #                 'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
        #                 'accelerometer_link_6', 'velocimeter_link_6', 'gyro_link_6', 'magnetometer_link_6'],
        'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                        'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                        'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                        'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                        'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',],
        'lidar_max_dist': 3,
        'lidar_num_bins': 10,
        'lidar_num_bins3D': 6,
        'lidar_body': ['link_1', 'link_3', 'link_5'],
        'render_lidar_radius': 0.25,
        'hazard3Ds_num': 0,
        'vases_num': 0,
        'robot_locations':[(0.0,0.0)],
        'robot_rot':0
    }

    env = Engine(config)
    obs = env.reset()
    done = False
    ep_ret = 0
    ep_cost = 0
    cnt = 0
    T = 1000
    with open("link_pos.txt",'w+') as f:
        pos_6 = []
        np.savetxt(f, pos_6,  delimiter = ' ')
    while True:
        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f'%(ep_ret, ep_cost))
            ep_ret, ep_cost = 0, 0
            obs = env.reset()
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        # act = [0.0, 0.0]
        # act = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # cnt = cnt%T
        # print(cnt)
        # cnt += 1
        # if cnt  > 100:
        #     act = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  
        # if cnt > 200:
        #     act = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # if cnt > 300:
        #     act = [0.0, -0.3, 0.0, 0.0, 0.0, 0.0]
        # if cnt > 400:
        #     act = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # if cnt > 500:
        #     act = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # if cnt > 600:
        #     act = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # if cnt > 700:
        #     act = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        # if cnt > 800:
        #     act = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # if cnt > 900:
        #     act = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # act = [0.0, 0.0, 0.0]
        # if cnt  > 100:
        #     act = [0.5, 0.0, 0.0] 
        # if cnt > 800:
        #     act = [0.0, 0.0, 0.0]
        # if cnt > 300:
        #     act = [0.0, -0.5, 0.0, 0.0, 0.0, 0.0]
        # if cnt > 400:
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

        assert env.action_space.contains(act)
        obs, reward, done, info = env.step(act)
        # print(obs['accelerometer_link_2'])
        # joint = []
        # for i in range(6):
        #     joint.append(obs['jointpos_joint_'+str(i + 1)])
        # print(joint)
        # print('reward', reward)
        ep_ret += reward
        a = info['cost']
        ep_cost += info.get('cost', 0)
        # env.render()


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
