from typing_extensions import TypeGuard
from Cython.Compiler.Nodes import TryExceptStatNode
import safety_gym.envs

from safety_gym.envs.engine import Engine
from gym.envs.registration import register

# config_PointNavi = {
#     'robot_base': 'xmls/point.xml',
#     'task': 'goal',

#     'goal_size': 0.2,
#     'goal_keepout': 0.205,
#     'hazards_size': 0.2,
#     'hazards_keepout': 0.18,

#     # 'pillars_size': 0.15,
#     # 'pillars_keepout': 0.18,


#     'placements_extents': [-1.5, -1.5, 1.5, 1.5],
#     'hazards_num': 8,
#     #'pillars_num': 4,

#     'observe_goal_lidar': True,
#     'observe_hazards': True,  # Observe the vector from agent to hazards
#     'constrain_hazards': True,  # Constrain robot from being in hazardous areas
#     # 'observe_pillars': True,  
#     # 'constrain_pillars': True,  
#     'constrain_indicator': True,  # If true, all costs are either 1 or 0 for a given step.    
    
#     'reward_distance': 1.0, # Sparse reward

#     'lidar_max_dist': 3,
#     'lidar_num_bins': 16,
    
# }

# config_PointNavi = {

#         'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
#         'task': 'goal',
#         'observation_flatten': True,  # Flatten observation into a vector
#         'observe_sensors': True,  # Observe all sensor data from simulator
#         # Sensor observations
#         # Specify which sensors to add to observation space
#         'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
#         'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
#         'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
#         'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

#         #observe goal/box/...
#         'observe_goal_dist': False,  # Observe the distance to the goal
#         'observe_goal_comp': False,  # Observe a compass vector to the goal
#         'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
#         'observe_box_comp': False,  # Observe the box with a compass
#         'observe_box_lidar': False,  # Observe the box with a lidar
#         'observe_circle': False,  # Observe the origin with a lidar
#         'observe_remaining': False,  # Observe the fraction of steps remaining
#         'observe_walls': False,  # Observe the walls with a lidar space
#         'observe_hazards': True,  # Observe the vector from agent to hazards
#         'observe_vases': True,  # Observe the vector from agent to vases
#         'observe_pillars': False,  # Lidar observation of pillar object positions
#         'observe_buttons': False,  # Lidar observation of button object positions
#         'observe_gremlins': False,  # Gremlins are observed with lidar-like space
#         'observe_vision': False,  # Observe vision from the robot

#         # Constraints - flags which can be turned on
#         # By default, no constraints are enabled, and all costs are indicator functions.
#         'constrain_hazards': True,  # Constrain robot from being in hazardous areas
#         'constrain_vases': False,  # Constrain frobot from touching objects
#         'constrain_pillars': False,  # Immovable obstacles in the environment
#         'constrain_buttons': False,  # Penalize pressing incorrect buttons
#         'constrain_gremlins': False,  # Moving objects that must be avoided
#         # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
#         'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

#         #lidar setting
#         'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
#         'lidar_num_bins': 16,
#         #num setting
#         'hazards_num': 4,
#         'hazards_size': 0.30,
#         'vases_num': 0,



#         # Frameskip is the number of physics simulation steps per environment step
#         # Frameskip is sampled as a binomial distribution
#         # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
#         'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) 
#         'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
# }


# register(id='SafetyPointGoal-v0',
#          entry_point='safety_gym.envs.mujoco:Engine',
#          max_episode_steps=500,
#          kwargs={'config': config_PointNavi})
