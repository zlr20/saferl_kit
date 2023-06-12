#!/usr/bin/env python

import gym
import gym.spaces
import numpy as np
from PIL import Image
from copy import deepcopy
from collections import OrderedDict
import mujoco_py
from mujoco_py import MjViewer, MujocoException, const, MjRenderContextOffscreen

from safe_rl_envs.envs.world import World, Robot

import sys
from .engine_utils import *


# Distinct colors for different types of objects.
# For now this is mostly used for visualization.
# This also affects the vision observation, so if training from pixels.
COLOR_BOX = np.array([1, 1, 0, 1])
COLOR_BUTTON = np.array([1, .5, 0, 1])
COLOR_GOAL = np.array([0, 1, 0, 1])
COLOR_VASE = np.array([0, 1, 1, 1])
COLOR_HAZARD = np.array([0, 0, 1, 1])
COLOR_HAZARD3D = np.array([0, 0, 1, 1])
COLOR_PILLAR = np.array([.5, .5, 1, 1])
COLOR_WALL = np.array([.5, .5, .5, 1])
COLOR_GREMLIN = np.array([0.5, 0, 1, 1])
COLOR_CIRCLE = np.array([0, 1, 0, 1])
COLOR_RED = np.array([1, 0, 0, 1])
COLOR_GHOST = np.array([1, 0, 0.5, 1])
COLOR_GHOST3D = np.array([1, 0, 0.5, 1])
COLOR_ROBBER = np.array([1, 1, 0.5, 1])
COLOR_ROBBER3D = np.array([1, 1, 0.5, 1])

# Groups are a mujoco-specific mechanism for selecting which geom objects to "see"
# We use these for raycasting lidar, where there are different lidar types.
# These work by turning "on" the group to see and "off" all the other groups.
# See obs_lidar_natural() for more.
GROUP_GOAL = 0
GROUP_BOX = 1
GROUP_BUTTON = 1
GROUP_WALL = 2
GROUP_PILLAR = 2
GROUP_HAZARD = 3
GROUP_VASE = 4
GROUP_GREMLIN = 5
GROUP_CIRCLE = 6
GROUP_HAZARD3D = 3
GROUP_GHOST = 3
GROUP_GHOST3D = 3
GROUP_ROBBER = 5
GROUP_ROBBER3D = 5



# Constant for origin of world
ORIGIN_COORDINATES = np.zeros(3)

# Constant defaults for rendering frames for humans (not used for vision)
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

class ResamplingError(AssertionError):
    ''' Raised when we fail to sample a valid distribution of objects or goals '''
    pass


def theta2vec(theta):
    ''' Convert an angle (in radians) to a unit vector in that angle around Z '''
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def quat2mat(quat):
    ''' Convert Quaternion to a 3x3 Rotation Matrix using mujoco '''
    q = np.array(quat, dtype='float64')
    m = np.zeros(9, dtype='float64')
    mujoco_py.functions.mju_quat2Mat(m, q)
    return m.reshape((3,3))


def quat2zalign(quat):
    ''' From quaternion, extract z_{ground} dot z_{body} '''
    # z_{body} from quaternion [a,b,c,d] in ground frame is:
    # [ 2bd + 2ac,
    #   2cd - 2ab,
    #   a**2 - b**2 - c**2 + d**2
    # ]
    # so inner product with z_{ground} = [0,0,1] is
    # z_{body} dot z_{ground} = a**2 - b**2 - c**2 + d**2
    a, b, c, d = quat
    return a**2 - b**2 - c**2 + d**2


class Engine(gym.Env, gym.utils.EzPickle):

    '''
    Engine: an environment-building tool for safe exploration research.

    The Engine() class entails everything to do with the tasks and safety 
    requirements of Safety Gym environments. An Engine() uses a World() object
    to interface to MuJoCo. World() configurations are inferred from Engine()
    configurations, so an environment in Safety Gym can be completely specified
    by the config dict of the Engine() object.

    '''

    # Default configuration (this should not be nested since it gets copied)
    DEFAULT = {
        'num_steps': 1000,  # Maximum number of environment steps in an episode

        'action_noise': 0.0,  # Magnitude of independent per-component gaussian action noise

        'placements_extents': [-2, -2, 2, 2],  # Placement limits (min X, min Y, max X, max Y)
        'placements_margin': 0.0,  # Additional margin added to keepout when placing objects

        # Floor
        'floor_display_mode': False,  # In display mode, the visible part of the floor is cropped

        # Robot
        'robot_placements': None,  # Robot placements list (defaults to full extents)
        'robot_locations': [],  # Explicitly place robot XY coordinate
        'robot_keepout': 0.4,  # Needs to be set to match the robot XML used
        'robot_base': 'xmls/car.xml',  # Which robot XML to use as the base
        'robot_rot': None,  # Override robot starting angle

        # Starting position distribution
        'randomize_layout': True,  # If false, set the random seed before layout to constant
        'build_resample': True,  # If true, rejection sample from valid environments
        'continue_goal': True,  # If true, draw a new goal after achievement
        'terminate_resample_failure': True,  # If true, end episode when resampling fails,
                                             # otherwise, raise a python exception.
        # TODO: randomize starting joint positions

        # Observation flags - some of these require other flags to be on
        # By default, only robot sensor observations are enabled.
        'observation_flatten': True,  # Flatten observation into a vector
        'observe_sensors': True,  # Observe all sensor data from simulator
        'observe_goal_dist': False,  # Observe the distance to the goal
        'observe_goal_comp': False,  # Observe a compass vector to the goal
        'observe_goal_lidar': False,  # Observe the goal with a lidar sensor
        'observe_box_comp': False,  # Observe the box with a compass
        'observe_box_lidar': False,  # Observe the box with a lidar
        'observe_circle': False,  # Observe the origin with a lidar
        'observe_remaining': False,  # Observe the fraction of steps remaining
        'observe_walls': False,  # Observe the walls with a lidar space
        'observe_hazards': False,  # Observe the vector from agent to hazards
        'observe_hazard3Ds': False,  # Observe the vector from agent to hazards
        'observe_vases': False,  # Observe the vector from agent to vases
        'observe_pillars': False,  # Lidar observation of pillar object positions
        'observe_buttons': False,  # Lidar observation of button object positions
        'observe_gremlins': False,  # Gremlins are observed with lidar-like space
        'observe_ghosts': False,  # Ghosts are observed with lidar-like space
        'observe_ghost3Ds': False,  # Ghosts are observed with lidar-like space
        'observe_robbers': False,  # Robbers are observed with lidar-like space
        'observe_robber3Ds': False,  # Robbers are observed with lidar-like space
        'observe_vision': False,  # Observe vision from the robot
        # These next observations are unnormalized, and are only for debugging
        'observe_qpos': False,  # Observe the qpos of the world
        'observe_qvel': False,  # Observe the qvel of the robot
        'observe_ctrl': False,  # Observe the previous action
        'observe_freejoint': False,  # Observe base robot free joint
        'observe_com': False,  # Observe the center of mass of the robot

        # Render options
        'render_labels': False,
        'render_lidar_markers': True,
        'render_lidar_radius': 0.15, 
        'render_lidar_size': 0.025, 
        'render_lidar_offset_init': 0.5, 
        'render_lidar_offset_delta': 0.06, 

        # Vision observation parameters
        'vision_size': (60, 40),  # Size (width, height) of vision observation; gets flipped internally to (rows, cols) format
        'vision_render': True,  # Render vision observation in the viewer
        'vision_render_size': (300, 200),  # Size to render the vision in the viewer

        # Lidar observation parameters
        'lidar_num_bins': 16,  # Bins (around a full circle) for lidar sensing
        'lidar_num_bins3D': 1,  # Bins (around a full circle) for lidar sensing
        'lidar_max_dist': None,  # Maximum distance for lidar sensitivity (if None, exponential distance)
        'lidar_exp_gain': 1.0, # Scaling factor for distance in exponential distance lidar
        'lidar_type': 'pseudo',  # 'pseudo', 'natural', see self.obs_lidar()
        'lidar_alias': True,  # Lidar bins alias into each other
        'lidar_body': ['robot'], # Bodies with Lidar

        # Compass observation parameters
        'compass_shape': 2,  # Set to 2 or 3 for XY or XYZ unit vector compass observation.

        # Task
        'task': 'goal',  # goal, button, push, x, z, circle, or none (for screenshots)
        'push_object': 'box', # box, ball
        'goal_mode': 'random', # random, track. only apply when continue_goal is true
        'goal_travel': 3.0,  # Radius of the circle goal can travel in
        'goal_velocity':0.5, # Velocity of the goal under 'track' mode

        # Goal parameters
        'goal_placements': None,  # Placements where goal may appear (defaults to full extents)
        'goal_locations': [],  # Fixed locations to override placements
        'goal_keepout': 0.4,  # Keepout radius when placing goals
        'goal_size': 0.3,  # Radius of the goal area (if using task 'goal')
        'goal_3D': False,
        'goal_z_range': [1.0, 1.0],  # range of z pos of goal, only for 3D goal

        # Box parameters (only used if task == 'push')
        'box_placements': None,  # Box placements list (defaults to full extents)
        'box_locations': [],  # Fixed locations to override placements
        'box_keepout': 0.2,  # Box keepout radius for placement
        'box_size': 0.2,  # Box half-radius size
        'box_density': 0.001,  # Box density
        'box_null_dist': 2, # Within box_null_dist * box_size radius of box, no box reward given

        # Reward is distance towards goal plus a constant for being within range of goal
        # reward_distance should be positive to encourage moving towards the goal
        # if reward_distance is 0, then the reward function is sparse
        'reward_distance': 1.0,  # Dense reward multiplied by the distance moved to the goal
        'reward_goal': 1.0,  # Sparse reward for being inside the goal area
        'reward_box_dist': 1.0,  # Dense reward for moving the robot towards the box
        'reward_box_goal': 1.0,  # Reward for moving the box towards the goal
        'reward_orientation': False,  # Reward for being upright
        'reward_orientation_scale': 0.002,  # Scale for uprightness reward
        'reward_orientation_body': 'robot',  # What body to get orientation from
        'reward_exception': -10.0,  # Reward when encoutering a mujoco exception
        'reward_x': 1.0,  # Reward for forward locomotion tests (vel in x direction)
        'reward_z': 1.0,  # Reward for standup tests (vel in z direction)
        'reward_circle': 1e-1,  # Reward for circle goal (complicated formula depending on pos and vel)
        'reward_clip': 10,  # Clip reward, last resort against physics errors causing magnitude spikes
        'reward_defense': 1.0, # Reward for the robbers be outside of the circle
        'reward_chase': 1.0, # Reward for the closest distance from the robot to the robbers

        # Buttons are small immovable spheres, to the environment
        'buttons_num': 0,  # Number of buttons to add
        'buttons_placements': None,  # Buttons placements list (defaults to full extents)
        'buttons_locations': [],  # Fixed locations to override placements
        'buttons_keepout': 0.3,  # Buttons keepout radius for placement
        'buttons_size': 0.1,  # Size of buttons in the scene
        'buttons_cost': 1.0,  # Cost for pressing the wrong button, if constrain_buttons
        'buttons_resampling_delay': 10,  # Buttons have a timeout period (steps) before resampling

        # Circle parameters (only used if task == 'circle' or 'defense')
        'circle_radius': 1.5,
        'defense_range': 2.0,
        'chase_range': 3.0,

        # Sensor observations
        # Specify which sensors to add to observation space
        'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
        'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
        'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
        'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

        # Walls - barriers in the environment not associated with any constraint
        # NOTE: this is probably best to be auto-generated than manually specified
        'walls_num': 0,  # Number of walls
        'walls_placements': None,  # This should not be used
        'walls_locations': [],  # This should be used and length == walls_num
        'walls_keepout': 0.0,  # This should not be used
        'walls_size': 0.5,  # Should be fixed at fundamental size of the world

        # Constraints - flags which can be turned on
        # By default, no constraints are enabled, and all costs are indicator functions.
        'constrain_hazards': False,  # Constrain robot from being in hazardous areas
        'constrain_hazard3Ds': False,  # Constrain robot from being in 3D hazardous areas
        'constrain_vases': False,  # Constrain frobot from touching objects
        'constrain_pillars': False,  # Immovable obstacles in the environment
        'constrain_buttons': False,  # Penalize pressing incorrect buttons
        'constrain_gremlins': False,  # Moving objects that must be avoided
        'constrain_ghosts': False,  # Moving objects that must be avoided
        'constrain_ghost3Ds': False,  # Moving objects that must be avoided
        'constrain_indicator': True,  # If true, all costs are either 1 or 0 for a given step.

        # Hazardous areas
        'hazards_num': 0,  # Number of hazards in an environment
        'hazards_placements': None,  # Placements list for hazards (defaults to full extents)
        'hazards_locations': [],  # Fixed locations to override placements
        'hazards_keepout': 0.4,  # Radius of hazard keepout for placement
        'hazards_size': 0.3,  # Radius of hazards
        'hazards_cost': 1.0,  # Cost (per step) for violating the constraint

        # 3D Hazardous areas
        'hazard3Ds_num': 0,  # Number of hazards in an environment
        'hazard3Ds_placements': None,  # Placements list for hazards (defaults to full extents)
        'hazard3Ds_locations': [],  # Fixed locations to override placements
        'hazard3Ds_keepout': 0.4,  # Radius of hazard keepout for placement
        'hazard3Ds_size': 0.3,  # Radius of hazards
        'hazard3Ds_z': [],  # z pos of hazards
        'hazard3Ds_z_range': [1.0, 1.0],  # range of z pos
        'hazard3Ds_cost': 1.0,  # Cost (per step) for violating the constraint

        # Vases (objects we should not touch)
        'vases_num': 0,  # Number of vases in the world
        'vases_placements': None,  # Vases placements list (defaults to full extents)
        'vases_locations': [],  # Fixed locations to override placements
        'vases_keepout': 0.15,  # Radius of vases keepout for placement
        'vases_size': 0.1,  # Half-size (radius) of vase object
        'vases_density': 0.01,  # Density of vases
        'vases_sink': 4e-5,  # Experimentally measured, based on size and density,
                             # how far vases "sink" into the floor.
        # Mujoco has soft contacts, so vases slightly sink into the floor,
        # in a way which can be hard to precisely calculate (and varies with time)
        # Ignore some costs below a small threshold, to reduce noise.
        'vases_contact_cost': 1.0,  # Cost (per step) for being in contact with a vase
        'vases_displace_cost': 0.0,  # Cost (per step) per meter of displacement for a vase
        'vases_displace_threshold': 1e-3,  # Threshold for displacement being "real"
        'vases_velocity_cost': 1.0,  # Cost (per step) per m/s of velocity for a vase
        'vases_velocity_threshold': 1e-4,  # Ignore very small velocities

        # Pillars (immovable obstacles we should not touch)
        'pillars_num': 0,  # Number of pillars in the world
        'pillars_placements': None,  # Pillars placements list (defaults to full extents)
        'pillars_locations': [],  # Fixed locations to override placements
        'pillars_keepout': 0.3,  # Radius for placement of pillars
        'pillars_size': 0.2,  # Half-size (radius) of pillar objects
        'pillars_height': 0.5,  # Half-height of pillars geoms
        'pillars_cost': 1.0,  # Cost (per step) for being in contact with a pillar

        # Gremlins (moving objects we should avoid)
        'gremlins_num': 0,  # Number of gremlins in the world
        'gremlins_placements': None,  # Gremlins placements list (defaults to full extents)
        'gremlins_locations': [],  # Fixed locations to override placements
        'gremlins_keepout': 0.5,  # Radius for keeping out (contains gremlin path)
        'gremlins_travel': 0.3,  # Radius of the circle traveled in
        'gremlins_size': 0.1,  # Half-size (radius) of gremlin objects
        'gremlins_density': 0.001,  # Density of gremlins
        'gremlins_contact_cost': 1.0,  # Cost for touching a gremlin
        'gremlins_dist_threshold': 0.2,  # Threshold for cost for being too close
        'gremlins_dist_cost': 1.0,  # Cost for being within distance threshold

        # Ghosts (hazards moving towards / away from the robot)
        'ghosts_num': 0,  # Number of ghosts in the world
        'ghosts_placements': None,  # Ghosts placements list (defaults to full extents)
        'ghosts_locations': [],  # Fixed locations to override placements
        'ghosts_keepout': 0.5,  # Radius for keeping out
        'ghosts_travel': 2,  # Radius of the circle traveled in
        'ghosts_size': 0.1,  # Half-size (radius) of ghost objects
        'ghosts_density': 0.00001,  # Density of ghosts
        'ghosts_contact_cost': 1.0,  # Cost for touching a gremlin
        'ghosts_dist_cost': 1.0,  # Cost for being within distance threshold
        'ghosts_velocity': 0.0001,
        'ghosts_contact': False,
        'ghosts_safe_dist': 0,

        # 3D Ghosts (hazards moving towards / away from the robot)
        'ghost3Ds_num': 0,  # Number of ghosts in the world
        'ghost3Ds_placements': None,  # Ghosts placements list (defaults to full extents)
        'ghost3Ds_locations': [],  # Fixed locations to override placements
        'ghost3Ds_keepout': 0.5,  # Radius for keeping out
        'ghost3Ds_travel': 2,  # Radius of the circle traveled in
        'ghost3Ds_size': 0.1,  # Half-size (radius) of ghost objects
        'ghost3Ds_density': 1,  # Density of ghosts
        'ghost3Ds_contact_cost': 1.0,  # Cost for touching a ghost
        'ghost3Ds_dist_cost': 1.0,  # Cost for being within distance threshold
        'ghost3Ds_velocity': 0.0001,
        'ghost3Ds_contact': False,
        'ghost3Ds_z_range': [0.0, 0.0],
        'ghost3Ds_safe_dist': 0.1,

        # Robbers (targets moving away from the robot)
        'robbers_num': 0,  # Number of robbers in the world
        'robbers_placements': None,  # Robbers placements list (defaults to full extents)
        'robbers_locations': [],  # Fixed locations to override placements
        'robbers_keepout': 0.5,  # Radius for keeping out
        'robbers_travel': 2,  # Radius of the circle traveled in
        'robbers_size': 0.1,  # Half-size (radius) of robber objects
        'robbers_density': 0.00001,  # Density of robbers
        'robbers_velocity': 0.0001,
        'robbers_contact': False,

        # 3D Robbers (targets moving away from the robot)
        'robber3Ds_num': 0,  # Number of robbers in the world
        'robber3Ds_placements': None,  # Robbers placements list (defaults to full extents)
        'robber3Ds_locations': [],  # Fixed locations to override placements
        'robber3Ds_keepout': 0.5,  # Radius for keeping out
        'robber3Ds_travel': 2,  # Radius of the circle traveled in
        'robber3Ds_size': 0.1,  # Half-size (radius) of robber objects
        'robber3Ds_density': 1,  # Density of robbers
        'robber3Ds_velocity': 0.0001,
        'robber3Ds_contact': False,
        'robber3Ds_z_range': [0.0, 0.0],

        # Frameskip is the number of physics simulation steps per environment step
        # Frameskip is sampled as a binomial distribution
        # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
        'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip)
        'frameskip_binom_p': 1.0,  # Probability of trial return (controls distribution)

        # For robot arm
        'arm_link_n': 7,

        '_seed': None,  # Random state seed (avoid name conflict with self.seed)
    }

    def __init__(self, config={}):
        # First, parse configuration. Important note: LOTS of stuff happens in
        # parse, and many attributes of the class get set through setattr. If you
        # are trying to track down where an attribute gets initially set, and 
        # can't find it anywhere else, it's probably set via the config dict
        # and this parse function.
        self.parse(config)
        gym.utils.EzPickle.__init__(self, config=config)

        # Load up a simulation of the robot, just to figure out observation space
        self.robot = Robot(self.robot_base)
        if 'arm_6' in self.robot_base:
            self.arm_link_n = 7
        if 'arm_3' in self.robot_base:
            self.arm_link_n = 5

        self.action_space = gym.spaces.Box(-1, 1, (self.robot.nu,), dtype=np.float32)
        self.build_observation_space()
        self.build_placements_dict()

        self.viewer = None
        self.world = None
        self.clear()

        self.seed(self._seed)
        self.done = True

    def parse(self, config):
        ''' Parse a config dict - see self.DEFAULT for description '''
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

    @property
    def sim(self):
        ''' Helper to get the world's simulation instance '''
        return self.world.sim

    @property
    def model(self):
        ''' Helper to get the world's model instance '''
        return self.sim.model

    @property
    def data(self):
        ''' Helper to get the world's simulation data instance '''
        return self.sim.data

    @property
    def robot_pos(self):
        ''' Helper to get current robot position '''
        return self.data.get_body_xpos('robot').copy()
    
    @property
    def arm_end_pos(self):
        ''' Helper to get current robot position '''
        return self.data.get_body_xpos('link_' + str(self.arm_link_n)).copy()

    @property
    def goal_pos(self):
        ''' Helper to get goal position from layout '''
        if self.task in ['goal', 'push']:
            return self.data.get_body_xpos('goal').copy()
        elif self.task == 'button':
            return self.data.get_body_xpos(f'button{self.goal_button}').copy()
        elif self.task == 'circle':
            return ORIGIN_COORDINATES
        elif self.task == 'none':
            return np.zeros(2)  # Only used for screenshots
        elif self.task == 'chase':
            return ORIGIN_COORDINATES
        elif self.task == 'defense':
            return ORIGIN_COORDINATES
        else:
            raise ValueError(f'Invalid task {self.task}')

    @property
    def box_pos(self):
        ''' Helper to get the box position '''
        return self.data.get_body_xpos('box').copy()

    @property
    def buttons_pos(self):
        ''' Helper to get the list of button positions '''
        return [self.data.get_body_xpos(f'button{i}').copy() for i in range(self.buttons_num)]

    @property
    def vases_pos(self):
        ''' Helper to get the list of vase positions '''
        return [self.data.get_body_xpos(f'vase{p}').copy() for p in range(self.vases_num)]

    @property
    def gremlins_obj_pos(self):
        ''' Helper to get the current gremlin position '''
        return [self.data.get_body_xpos(f'gremlin{i}obj').copy() for i in range(self.gremlins_num)]
    
    @property
    def ghosts_pos(self):
        ''' Helper to get the current ghost position '''
        if self.ghosts_contact:
            return [self.data.get_body_xpos(f'ghost{i}obj').copy() for i in range(self.ghosts_num)]
        else:
            return [self.data.get_body_xpos(f'ghost{i}mocap').copy()  + np.r_[self.layout[f'ghost{i}'], 2e-2] for i in range(self.ghosts_num)]
    
    @property
    def ghost3Ds_pos(self):
        ''' Helper to get the current 3D ghost position '''
        if self.ghost3Ds_contact:
            return [self.data.get_body_xpos(f'ghost3D{i}obj').copy() for i in range(self.ghost3Ds_num)]
        else:
            return [self.data.get_body_xpos(f'ghost3D{i}mocap').copy() + np.r_[self.layout[f'ghost3D{i}'],self._ghost3Ds_z[i]] for i in range(self.ghost3Ds_num)]

    @property
    def robbers_pos(self):
        ''' Helper to get the current robber position '''
        if self.robbers_contact:
            return [self.data.get_body_xpos(f'robber{i}obj').copy() for i in range(self.robbers_num)]
        else:
            return [self.data.get_body_xpos(f'robber{i}mocap').copy()  + np.r_[self.layout[f'robber{i}'], 2e-2] for i in range(self.robbers_num)]
    
    @property
    def robber3Ds_pos(self):
        ''' Helper to get the current 3D robber position '''
        if self.robber3Ds_contact:
            return [self.data.get_body_xpos(f'robber3D{i}obj').copy() for i in range(self.robber3Ds_num)]
        else:
            return [self.data.get_body_xpos(f'robber3D{i}mocap').copy() + np.r_[self.layout[f'robber3D{i}'],self._robber3Ds_z[i]] for i in range(self.robber3Ds_num)]

    @property
    def pillars_pos(self):
        ''' Helper to get list of pillar positions '''
        return [self.data.get_body_xpos(f'pillar{i}').copy() for i in range(self.pillars_num)]

    @property
    def hazards_pos(self):
        ''' Helper to get the hazards positions from layout '''
        return [self.data.get_body_xpos(f'hazard{i}').copy() for i in range(self.hazards_num)]
    
    @property
    def hazard3Ds_pos(self):
        ''' Helper to get the hazards positions from layout '''
        return [self.data.get_body_xpos(f'hazard3D{i}').copy() for i in range(self.hazard3Ds_num)]

    @property
    def walls_pos(self):
        ''' Helper to get the hazards positions from layout '''
        return [self.data.get_body_xpos(f'wall{i}').copy() for i in range(self.walls_num)]

    def build_observation_space(self):
        ''' Construct observtion space.  Happens only once at during __init__ '''
        obs_space_dict = OrderedDict()  # See self.obs()

        if self.observe_freejoint:
            obs_space_dict['freejoint'] = gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)
        if self.observe_com:
            obs_space_dict['com'] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        if self.observe_sensors:
            for sensor in self.sensors_obs:  # Explicitly listed sensors
                dim = self.robot.sensor_dim[sensor]
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
            # Velocities don't have wraparound effects that rotational positions do
            # Wraparounds are not kind to neural networks
            # Whereas the angle 2*pi is very close to 0, this isn't true in the network
            # In theory the network could learn this, but in practice we simplify it
            # when the sensors_angle_components switch is enabled.
            for sensor in self.robot.hinge_vel_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
            for sensor in self.robot.ballangvel_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
            # Angular positions have wraparound effects, so output something more friendly
            if self.sensors_angle_components:
                # Single joints are turned into sin(x), cos(x) pairs
                # These should be easier to learn for neural networks,
                # Since for angles, small perturbations in angle give small differences in sin/cos
                for sensor in self.robot.hinge_pos_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
                # Quaternions are turned into 3x3 rotation matrices
                # Quaternions have a wraparound issue in how they are normalized,
                # where the convention is to change the sign so the first element to be positive.
                # If the first element is close to 0, this can mean small differences in rotation
                # lead to large differences in value as the latter elements change sign.
                # This also means that the first element of the quaternion is not expectation zero.
                # The SO(3) rotation representation would be a good replacement here,
                # since it smoothly varies between values in all directions (the property we want),
                # but right now we have very little code to support SO(3) roatations.
                # Instead we use a 3x3 rotation matrix, which if normalized, smoothly varies as well.
                for sensor in self.robot.ballquat_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (3, 3), dtype=np.float32)
            else:
                # Otherwise include the sensor without any processing
                # TODO: comparative study of the performance with and without this feature.
                for sensor in self.robot.hinge_pos_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
                for sensor in self.robot.ballquat_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
        if self.task == 'push':
            if self.observe_box_comp:
                obs_space_dict['box_compass'] = gym.spaces.Box(-1.0, 1.0, (len(self.lidar_body), self.compass_shape), dtype=np.float32)
            if self.observe_box_lidar:
                obs_space_dict['box_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_goal_dist:
            obs_space_dict['goal_dist'] = gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        if self.observe_goal_comp:
            obs_space_dict['goal_compass'] = gym.spaces.Box(-1.0, 1.0, (len(self.lidar_body),self.compass_shape), dtype=np.float32)
        if self.observe_goal_lidar:
            if self.goal_3D:
                obs_space_dict['goal_lidar'] = gym.spaces.Box(0.0, 1.0, (len(self.lidar_body), self.lidar_num_bins, self.lidar_num_bins3D), dtype=np.float32)
            else:
                obs_space_dict['goal_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.task == 'circle' and self.observe_circle:
            obs_space_dict['circle_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_remaining:
            obs_space_dict['remaining'] = gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        if self.walls_num and self.observe_walls:
            obs_space_dict['walls_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_hazards:
            obs_space_dict['hazards_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_hazard3Ds:
            obs_space_dict['hazard3Ds_lidar'] = gym.spaces.Box(0.0, 1.0, (len(self.lidar_body), self.lidar_num_bins, self.lidar_num_bins3D), dtype=np.float32)
        if self.observe_vases:
            obs_space_dict['vases_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.gremlins_num and self.observe_gremlins:
            obs_space_dict['gremlins_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.ghosts_num and self.observe_ghosts:
            obs_space_dict['ghosts_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.ghost3Ds_num and self.observe_ghost3Ds:
            obs_space_dict['ghost3Ds_lidar'] = gym.spaces.Box(0.0, 1.0, (len(self.lidar_body), self.lidar_num_bins, self.lidar_num_bins3D), dtype=np.float32)
        if self.robbers_num and self.observe_robbers:
            obs_space_dict['robbers_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.robber3Ds_num and self.observe_robber3Ds:
            obs_space_dict['robber3Ds_lidar'] = gym.spaces.Box(0.0, 1.0, (len(self.lidar_body), self.lidar_num_bins, self.lidar_num_bins3D), dtype=np.float32)
        if self.pillars_num and self.observe_pillars:
            obs_space_dict['pillars_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.buttons_num and self.observe_buttons:
            obs_space_dict['buttons_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_qpos:
            obs_space_dict['qpos'] = gym.spaces.Box(-np.inf, np.inf, (self.robot.nq,), dtype=np.float32)
        if self.observe_qvel:
            obs_space_dict['qvel'] = gym.spaces.Box(-np.inf, np.inf, (self.robot.nv,), dtype=np.float32)
        if self.observe_ctrl:
            obs_space_dict['ctrl'] = gym.spaces.Box(-np.inf, np.inf, (self.robot.nu,), dtype=np.float32)
        if self.observe_vision:
            width, height = self.vision_size
            rows, cols = height, width
            self.vision_size = (rows, cols)
            obs_space_dict['vision'] = gym.spaces.Box(0, 1.0, self.vision_size + (3,), dtype=np.float32)
        # Flatten it ourselves
        self.obs_space_dict = obs_space_dict
        if self.observation_flatten:
            self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Dict(obs_space_dict)

    def toggle_observation_space(self):
        self.observation_flatten = not(self.observation_flatten)
        self.build_observation_space()

    def placements_from_location(self, location, keepout):
        ''' Helper to get a placements list from a given location and keepout '''
        x, y = location
        return [(x - keepout, y - keepout, x + keepout, y + keepout)]

    def placements_dict_from_object(self, object_name):
        ''' Get the placements dict subset just for a given object name '''
        placements_dict = {}
        if hasattr(self, object_name + 's_num'):  # Objects with multiplicity
            plural_name = object_name + 's'
            object_fmt = object_name + '{i}'
            object_num = getattr(self, plural_name + '_num', None)
            object_locations = getattr(self, plural_name + '_locations', [])
            object_placements = getattr(self, plural_name + '_placements', None)
            object_keepout = getattr(self, plural_name + '_keepout')
        else:  # Unique objects
            object_fmt = object_name
            object_num = 1
            object_locations = getattr(self, object_name + '_locations', [])
            object_placements = getattr(self, object_name + '_placements', None)
            object_keepout = getattr(self, object_name + '_keepout')
        for i in range(object_num):
            if i < len(object_locations):
                x, y = object_locations[i]
                k = object_keepout + 1e-9  # Epsilon to account for numerical issues
                placements = [(x - k, y - k, x + k, y + k)]
            else:
                placements = object_placements
            placements_dict[object_fmt.format(i=i)] = (placements, object_keepout)
        return placements_dict

    def build_placements_dict(self):
        ''' Build a dict of placements.  Happens once during __init__. '''
        # Dictionary is map from object name -> tuple of (placements list, keepout)
        placements = {}

        placements.update(self.placements_dict_from_object('robot'))
        placements.update(self.placements_dict_from_object('wall'))

        if self.task in ['goal', 'push']:
            placements.update(self.placements_dict_from_object('goal'))
        if self.task == 'push':
            placements.update(self.placements_dict_from_object('box'))
        if self.task == 'button' or self.buttons_num: #self.constrain_buttons:
            placements.update(self.placements_dict_from_object('button'))
        if self.hazards_num: #self.constrain_hazards:
            placements.update(self.placements_dict_from_object('hazard'))
        if self.hazard3Ds_num: #self.constrain_hazard3Ds:
            placements.update(self.placements_dict_from_object('hazard3D'))
        if self.vases_num: #self.constrain_vases:
            placements.update(self.placements_dict_from_object('vase'))
        if self.pillars_num: #self.constrain_pillars:
            placements.update(self.placements_dict_from_object('pillar'))
        if self.gremlins_num: #self.constrain_gremlins:
            placements.update(self.placements_dict_from_object('gremlin'))
        if self.ghosts_num: #self.constrain_ghosts:
            placements.update(self.placements_dict_from_object('ghost'))
        if self.ghost3Ds_num: #self.constrain_ghosts:
            placements.update(self.placements_dict_from_object('ghost3D'))
        if self.robbers_num: #self.constrain_robbers:
            placements.update(self.placements_dict_from_object('robber'))
        if self.robber3Ds_num: #self.constrain_robbers:
            placements.update(self.placements_dict_from_object('robber3D'))

        self.placements = placements

    def seed(self, seed=None):
        ''' Set internal random state seeds '''
        self._seed = np.random.randint(2**32) if seed is None else seed

    def build_layout(self):
        ''' Rejection sample a placement of objects to find a layout. '''
        if not self.randomize_layout:
            self.rs = np.random.RandomState(0)

        for _ in range(10000):
            if self.sample_layout():
                break
        else:
            raise ResamplingError('Failed to sample layout of objects')

    def sample_layout(self):
        ''' Sample a single layout, returning True if successful, else False. '''

        def placement_is_valid(xy, layout):
            for other_name, other_xy in layout.items():
                other_keepout = self.placements[other_name][1]
                dist = np.sqrt(np.sum(np.square(xy - other_xy)))
                if dist < other_keepout + self.placements_margin + keepout:
                    return False
            return True

        layout = {}
        for name, (placements, keepout) in self.placements.items():
            conflicted = True
            for _ in range(100):
                xy = self.draw_placement(placements, keepout)
                if 'arm' in self.robot_base and 'hazard3D' in name and np.sqrt(np.sum(np.square(xy))) > 2:
                    continue
                if placement_is_valid(xy, layout):
                    conflicted = False
                    break
            if conflicted:
                return False
            layout[name] = xy
        
        self.layout = layout
        return True

    def constrain_placement(self, placement, keepout):
        ''' Helper function to constrain a single placement by the keepout radius '''
        xmin, ymin, xmax, ymax = placement
        return (xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout)

    def draw_placement(self, placements, keepout):
        ''' 
        Sample an (x,y) location, based on potential placement areas.

        Summary of behavior: 

        'placements' is a list of (xmin, xmax, ymin, ymax) tuples that specify 
        rectangles in the XY-plane where an object could be placed. 

        'keepout' describes how much space an object is required to have
        around it, where that keepout space overlaps with the placement rectangle.

        To sample an (x,y) pair, first randomly select which placement rectangle
        to sample from, where the probability of a rectangle is weighted by its
        area. If the rectangles are disjoint, there's an equal chance the (x,y) 
        location will wind up anywhere in the placement space. If they overlap, then
        overlap areas are double-counted and will have higher density. This allows
        the user some flexibility in building placement distributions. Finally, 
        randomly draw a uniform point within the selected rectangle.

        '''
        if placements is None:
            choice = self.constrain_placement(self.placements_extents, keepout)
        else:
            # Draw from placements according to placeable area
            constrained = []
            for placement in placements:
                xmin, ymin, xmax, ymax = self.constrain_placement(placement, keepout)
                if xmin > xmax or ymin > ymax:
                    continue
                constrained.append((xmin, ymin, xmax, ymax))
            assert len(constrained), 'Failed to find any placements with satisfy keepout'
            if len(constrained) == 1:
                choice = constrained[0]
            else:
                areas = [(x2 - x1)*(y2 - y1) for x1, y1, x2, y2 in constrained]
                probs = np.array(areas) / np.sum(areas)
                choice = constrained[self.rs.choice(len(constrained), p=probs)]
        xmin, ymin, xmax, ymax = choice
        return np.array([self.rs.uniform(xmin, xmax), self.rs.uniform(ymin, ymax)])

    def random_rot(self):
        ''' Use internal random state to get a random rotation in radians '''
        return self.rs.uniform(0, 2 * np.pi)

    def build_world_config(self):
        ''' Create a world_config from our own config '''
        # TODO: parse into only the pieces we want/need
        world_config = {}

        world_config['robot_base'] = self.robot_base
        world_config['robot_xy'] = self.layout['robot']
        if self.robot_rot is None:
            world_config['robot_rot'] = self.random_rot()
        else:
            world_config['robot_rot'] = float(self.robot_rot)

        if self.floor_display_mode:
            floor_size = max(self.placements_extents)
            world_config['floor_size'] = [floor_size + .1, floor_size + .1, 1]

        #if not self.observe_vision:
        #    world_config['render_context'] = -1  # Hijack this so we don't create context
        world_config['observe_vision'] = self.observe_vision

        # Extra objects to add to the scene
        world_config['objects'] = {}
        if self.vases_num:
            for i in range(self.vases_num):
                name = f'vase{i}'
                object = {'name': name,
                          'size': np.ones(3) * self.vases_size,
                          'type': 'box',
                          'density': self.vases_density,
                          'pos': np.r_[self.layout[name], self.vases_size - self.vases_sink],
                          'rot': self.random_rot(),
                          'group': GROUP_VASE,
                          'rgba': COLOR_VASE}
                world_config['objects'][name] = object
        if self.gremlins_num:
            self._gremlins_rots = dict()
            for i in range(self.gremlins_num):
                name = f'gremlin{i}obj'
                self._gremlins_rots[i] = self.random_rot()
                object = {'name': name,
                          'size': np.ones(3) * self.gremlins_size,
                          'type': 'box',
                          'density': self.gremlins_density,
                          'pos': np.r_[self.layout[name.replace('obj', '')], self.gremlins_size],
                          'rot': self._gremlins_rots[i],
                          'group': GROUP_GREMLIN,
                          'rgba': COLOR_GREMLIN}
                world_config['objects'][name] = object
        if self.ghosts_num:
            self._ghosts_rots = dict()
            for i in range(self.ghosts_num):
                name = f'ghost{i}obj'
                self._ghosts_rots[i] = self.random_rot()
                if self.ghosts_contact:
                    object = {'name': name,
                            'size': [self.ghosts_size, self.ghosts_size / 2],
                            'type': 'cylinder',
                            'density': self.ghosts_density,
                            'pos': np.r_[self.layout[name.replace('obj', '')], self.ghosts_size / 2 + 1e-2],
                            'rot': self._ghosts_rots[i],
                            'group': GROUP_GHOST,
                            'rgba': COLOR_GHOST}
                    world_config['objects'][name] = object
        if self.ghost3Ds_num:
            self._ghost3Ds_rots = dict()
            self._ghost3Ds_z = dict()
            for i in range(self.ghost3Ds_num):
                name = f'ghost3D{i}obj'
                self._ghost3Ds_rots[i] = self.random_rot()
                self._ghost3Ds_z[i] = np.random.uniform(self.ghost3Ds_z_range[0], self.ghost3Ds_z_range[1])
                if self.ghost3Ds_contact:
                    object = {'name': name,
                            'size': [self.ghost3Ds_size],
                            'type': 'sphere',
                            'density': self.ghost3Ds_density,
                            'pos': np.r_[self.layout[name.replace('obj', '')], self._ghost3Ds_z[i]],
                            'rot': self._ghost3Ds_rots[i],
                            'group': GROUP_GHOST3D,
                            'rgba': COLOR_GHOST3D}
                    world_config['objects'][name] = object
        if self.robbers_num:
            self._robbers_rots = dict()
            for i in range(self.robbers_num):
                name = f'robber{i}obj'
                self._robbers_rots[i] = self.random_rot()
                if self.robbers_contact:
                    object = {'name': name,
                            'size': [self.robbers_size, self.robbers_size / 2],
                            'type': 'cylinder',
                            'density': self.robbers_density,
                            'pos': np.r_[self.layout[name.replace('obj', '')], self.robbers_size / 2 + 1e-2],
                            'rot': self._robbers_rots[i],
                            'group': GROUP_ROBBER,
                            'rgba': COLOR_ROBBER}
                    world_config['objects'][name] = object
        if self.robber3Ds_num:
            self._robber3Ds_rots = dict()
            self._robber3Ds_z = dict()
            for i in range(self.robber3Ds_num):
                name = f'robber3D{i}obj'
                self._robber3Ds_rots[i] = self.random_rot()
                self._robber3Ds_z[i] = np.random.uniform(self.robber3Ds_z_range[0], self.robber3Ds_z_range[1])
                if self.robber3Ds_contact:
                    object = {'name': name,
                            'size': [self.robber3Ds_size],
                            'type': 'sphere',
                            'density': self.robber3Ds_density,
                            'pos': np.r_[self.layout[name.replace('obj', '')], self._robber3Ds_z[i]],
                            'rot': self._robber3Ds_rots[i],
                            'group': GROUP_ROBBER3D,
                            'rgba': COLOR_ROBBER3D}
                    world_config['objects'][name] = object
        if self.task == 'push':
            if self.push_object == 'box':
                object = {'name': 'box',
                        'type': 'box',
                        'size': np.ones(3) * self.box_size,
                        'pos': np.r_[self.layout['box'], self.box_size],
                        'rot': self.random_rot(),
                        'density': self.box_density,
                        'group': GROUP_BOX,
                        'rgba': COLOR_BOX}
            if self.push_object == 'ball':
                object = {'name': 'box',
                        'type': 'sphere',
                        'size': np.ones(3) * self.box_size,
                        'pos': np.r_[self.layout['box'], self.box_size],
                        'rot': self.random_rot(),
                        'density': self.box_density,
                        'group': GROUP_BOX,
                        'rgba': COLOR_BOX}
            world_config['objects']['box'] = object

        # Extra geoms (immovable objects) to add to the scene
        world_config['geoms'] = {}
        if self.task in ['goal', 'push']:
            if self.goal_3D == False:
                geom = {'name': 'goal',
                    'size': [self.goal_size, self.goal_size / 2],
                    'pos': np.r_[self.layout['goal'], self.goal_size / 2 + 1e-2],
                    'rot': self.random_rot(),
                    'type': 'cylinder',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': GROUP_GOAL,
                    'rgba': COLOR_GOAL * [1, 1, 1, 0.25]}  # transparent
            else:
                goal_pos = np.r_[self.layout['goal'], np.random.uniform(self.goal_z_range[0], self.goal_z_range[1])]
                geom = {'name': 'goal',
                    'size': [self.goal_size],
                    'pos': goal_pos,
                    'rot': self.random_rot(),
                    'type': 'sphere',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': GROUP_GOAL,
                    'rgba': COLOR_GOAL* [1, 1, 1, 0.25]}  # transparent * [1, 1, 1, 0.25]
            world_config['geoms']['goal'] = geom
        if self.hazards_num:
            for i in range(self.hazards_num):
                name = f'hazard{i}'
                geom = {'name': name,
                        'size': [self.hazards_size, 1e-2],#self.hazards_size / 2],
                        'pos': np.r_[self.layout[name], 2e-2],#self.hazards_size / 2 + 1e-2],
                        'rot': self.random_rot(),
                        'type': 'cylinder',
                        'contype': 0,
                        'conaffinity': 0,
                        'group': GROUP_HAZARD,
                        'rgba': COLOR_HAZARD * [1, 1, 1, 0.25]} #0.1]}  # transparent
                world_config['geoms'][name] = geom
        if self.hazard3Ds_num:
            for i in range(self.hazard3Ds_num):
                name = f'hazard3D{i}'
                if i < len(self.hazard3Ds_z):
                    pos_z = self.hazard3Ds_z[i]
                else:
                    pos_z = np.random.uniform(self.hazard3Ds_z_range[0], self.hazard3Ds_z_range[1])
                pos = np.r_[self.layout[name], pos_z]

                geom = {'name': name,
                        'size': [self.hazard3Ds_size],#self.hazards_size / 2],
                        'pos': pos,#self.hazards_size / 2 + 1e-2],
                        'rot': self.random_rot(),
                        'type': 'sphere',
                        'contype': 0,
                        'conaffinity': 0,
                        'group': GROUP_HAZARD3D,
                        'rgba': COLOR_HAZARD3D * [1, 1, 1, 0.25]} #0.1]}  # transparent
                world_config['geoms'][name] = geom
        if self.pillars_num:
            for i in range(self.pillars_num):
                name = f'pillar{i}'
                geom = {'name': name,
                        'size': [self.pillars_size, self.pillars_height],
                        'pos': np.r_[self.layout[name], self.pillars_height],
                        'rot': self.random_rot(),
                        'type': 'cylinder',
                        'group': GROUP_PILLAR,
                        'rgba': COLOR_PILLAR}
                world_config['geoms'][name] = geom
        if self.walls_num:
            for i in range(self.walls_num):
                name = f'wall{i}'
                geom = {'name': name,
                        'size': np.ones(3) * self.walls_size,
                        'pos': np.r_[self.layout[name], self.walls_size],
                        'rot': 0,
                        'type': 'box',
                        'group': GROUP_WALL,
                        'rgba': COLOR_WALL}
                world_config['geoms'][name] = geom
        if self.buttons_num:
            for i in range(self.buttons_num):
                name = f'button{i}'
                geom = {'name': name,
                        'size': np.ones(3) * self.buttons_size,
                        'pos': np.r_[self.layout[name], self.buttons_size],
                        'rot': self.random_rot(),
                        'type': 'sphere',
                        'group': GROUP_BUTTON,
                        'rgba': COLOR_BUTTON}
                world_config['geoms'][name] = geom
        if self.task in ['circle', 'defense']:
            geom = {'name': 'circle',
                    'size': np.array([self.circle_radius, 1e-2]),
                    'pos': np.array([0, 0, 2e-2]),
                    'rot': 0,
                    'type': 'cylinder',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': GROUP_CIRCLE,
                    'rgba': COLOR_CIRCLE * [1, 1, 1, 0.1]}
            world_config['geoms']['circle'] = geom


        # Extra mocap bodies used for control (equality to object of same name)
        world_config['mocaps'] = {}
        if self.gremlins_num:
            for i in range(self.gremlins_num):
                name = f'gremlin{i}mocap'
                mocap = {'name': name,
                         'size': np.ones(3) * self.gremlins_size,
                         'type': 'box',
                         'pos': np.r_[self.layout[name.replace('mocap', '')], self.gremlins_size],
                         'rot': self._gremlins_rots[i],
                         'group': GROUP_GREMLIN,
                         'rgba': np.array([1, 1, 1, .1]) * COLOR_GREMLIN}
                world_config['mocaps'][name] = mocap
        if self.ghosts_num:
            for i in range(self.ghosts_num):
                name = f'ghost{i}mocap'
                mocap = {'name': name,
                         'size': [self.ghosts_size, 1e-2],
                         'type': 'cylinder',
                         'pos': np.r_[self.layout[name.replace('mocap', '')], 2e-2], 
                         'rot': self._ghosts_rots[i],
                         'group': GROUP_GHOST,
                         'rgba': np.array([1, 1, 1, 0.25]) * COLOR_GHOST}
                world_config['mocaps'][name] = mocap
        if self.ghost3Ds_num:
            for i in range(self.ghost3Ds_num):
                name = f'ghost3D{i}mocap'
                if self.ghost3Ds_contact:
                    rgba = np.array([1, 1, 1, 0]) * COLOR_GHOST3D
                else:
                    rgba = np.array([1, 1, 1, 0.25]) * COLOR_GHOST3D
                mocap = {'name': name,
                         'size': [self.ghost3Ds_size],
                         'type': 'sphere',
                         'pos': np.r_[self.layout[name.replace('mocap', '')], self._ghost3Ds_z[i]], 
                         'rot': self._ghost3Ds_rots[i],
                         'group': GROUP_GHOST3D,
                         'rgba': rgba}
                world_config['mocaps'][name] = mocap
        if self.robbers_num:
            for i in range(self.robbers_num):
                name = f'robber{i}mocap'
                mocap = {'name': name,
                         'size': [self.robbers_size, 1e-2],
                         'type': 'cylinder',
                         'pos': np.r_[self.layout[name.replace('mocap', '')], 2e-2], 
                         'rot': self._robbers_rots[i],
                         'group': GROUP_ROBBER,
                         'rgba': np.array([1, 1, 1, 0.25]) * COLOR_ROBBER}
                world_config['mocaps'][name] = mocap
        if self.robber3Ds_num:
            for i in range(self.robber3Ds_num):
                name = f'robber3D{i}mocap'
                if self.robber3Ds_contact:
                    rgba = np.array([1, 1, 1, 0]) * COLOR_ROBBER3D
                else:
                    rgba = np.array([1, 1, 1, 0.25]) * COLOR_ROBBER3D
                mocap = {'name': name,
                         'size': [self.robber3Ds_size],
                         'type': 'sphere',
                         'pos': np.r_[self.layout[name.replace('mocap', '')], self._robber3Ds_z[i]], 
                         'rot': self._robber3Ds_rots[i],
                         'group': GROUP_ROBBER3D,
                         'rgba': rgba}
                world_config['mocaps'][name] = mocap

        return world_config

    def clear(self):
        ''' Reset internal state for building '''
        self.layout = None

    def build_goal(self):
        ''' Build a new goal position, maybe with resampling due to hazards '''
        if self.task == 'goal':
            self.build_goal_position()
            self.last_dist_goal = self.dist_goal()
        elif self.task == 'push':
            self.build_goal_position()
            self.last_dist_goal = self.dist_goal()
            self.last_dist_box = self.dist_box()
            self.last_box_goal = self.dist_box_goal()
        elif self.task == 'button':
            assert self.buttons_num > 0, 'Must have at least one button'
            self.build_goal_button()
            self.last_dist_goal = self.dist_goal()
        elif self.task in ['x', 'z']:
            self.last_robot_com = self.world.robot_com()
        elif self.task in ['circle', 'none', 'chase','defense']:
            pass
        else:
            raise ValueError(f'Invalid task {self.task}')

    def sample_goal_position(self):
        ''' Sample a new goal position and return True, else False if sample rejected '''
        placements, keepout = self.placements['goal']
        last_goal_pos = self.goal_pos
        if self.goal_mode == 'track':    
            if self.goal_3D == False:
                vec = np.random.normal(size=2)
                vec_norm = vec / np.linalg.norm(vec)
                direction = np.r_[vec_norm, 0]     
            else:
                vec = np.random.normal(size=3)
                vec_norm = vec / np.linalg.norm(vec)
                direction = vec_norm  
            goal_xyz = last_goal_pos + self.goal_velocity* direction
            for other_name, other_xy in self.layout.items():
                other_keepout = self.placements[other_name][1]
                dist = np.sqrt(np.sum(np.square(goal_xyz[:2] - other_xy)))
                if dist < other_keepout + self.placements_margin + keepout:
                    return False
            if 'arm' in self.robot_base:
                end_pos = self.arm_end_pos
                if np.sqrt(np.sum(np.square(goal_xyz - end_pos))) < self.goal_size:
                    return False
            goal_pos = goal_xyz
        else:  
            goal_xy = self.draw_placement(placements, keepout)
            for other_name, other_xy in self.layout.items():
                other_keepout = self.placements[other_name][1]
                dist = np.sqrt(np.sum(np.square(goal_xy - other_xy)))
                if dist < other_keepout + self.placements_margin + keepout:
                    return False
            goal_pos = np.r_[goal_xy, last_goal_pos[2]]

        if np.sqrt(np.sum(np.square(goal_pos[:2]))) > self.goal_travel:
            return False
        
        self.layout['goal'] = goal_pos
        return True

    def build_goal_position(self):
        ''' Build a new goal position, maybe with resampling due to hazards '''
        # Resample until goal is compatible with layout
        if 'goal' in self.layout:
            del self.layout['goal']
        for _ in range(10000):  # Retries
            if self.sample_goal_position():
                break
        else:
            raise ResamplingError('Failed to generate goal')
        # Move goal geom to new layout position
        self.world_config_dict['geoms']['goal']['pos'] = self.layout['goal']
        #self.world.rebuild(deepcopy(self.world_config_dict))
        #self.update_viewer_sim = True
        goal_body_id = self.sim.model.body_name2id('goal')
        self.sim.model.body_pos[goal_body_id] = self.layout['goal']
        self.sim.forward()
        self.layout['goal'] = self.layout['goal'][:2]

    def build_goal_button(self):
        ''' Pick a new goal button, maybe with resampling due to hazards '''
        self.goal_button = self.rs.choice(self.buttons_num)

    def build(self):
        ''' Build a new physics simulation environment '''
        # Sample object positions
        self.build_layout()

        # Build the underlying physics world
        self.world_config_dict = self.build_world_config()

        if self.world is None:
            self.world = World(self.world_config_dict)
            self.world.reset()
            self.world.build()
        else:
            self.world.reset(build=False)
            self.world.rebuild(self.world_config_dict, state=False)
        # Redo a small amount of work, and setup initial goal state
        self.build_goal()

        # Save last action
        self.last_action = np.zeros(self.action_space.shape)

        # Save last subtree center of mass
        self.last_subtreecom = self.world.get_sensor('subtreecom')

    def reset(self):
        ''' Reset the physics simulation and return observation '''
        self._seed += 1  # Increment seed
        self.rs = np.random.RandomState(self._seed)
        self.done = False
        self.steps = 0  # Count of steps taken in this episode
        # Set the button timer to zero (so button is immediately visible)
        self.buttons_timer = 0
        # self.last_dist_ghost = -1
        self.last_dist_robber = -1
        for _ in range(100):
            self.clear()
            self.build()
            cost = self.cost()
            if cost['cost'] == 0:
                break

        assert cost['cost'] == 0, f'World has starting cost! {cost}'

        # Save the layout at reset
        self.reset_layout = deepcopy(self.layout)

        
        

        # Reset stateful parts of the environment
        self.first_reset = False  # Built our first world successfully

        # Return an observation
        return self.obs()

    def dist_goal(self):
        ''' Return the distance from the robot to the goal XY position '''
        if 'arm' in self.robot_base:
            end_pos = self.arm_end_pos
            return np.sqrt(np.sum(np.square(self.goal_pos - end_pos)))
        if self.goal_3D:
             return self.dist_xyz(self.goal_pos)
        return self.dist_xy(self.goal_pos)

    def dist_box(self):
        ''' Return the distance from the robot to the box (in XY plane only) '''
        assert self.task == 'push', f'invalid task {self.task}'
        if 'arm' in self.robot_base:
            end_pos = self.arm_end_pos
            return np.sqrt(np.sum(np.square(self.box_pos - end_pos)))
        return np.sqrt(np.sum(np.square(self.box_pos - self.world.robot_pos())))

    def dist_box_goal(self):
        ''' Return the distance from the box to the goal XY position '''
        assert self.task == 'push', f'invalid task {self.task}'
        return np.sqrt(np.sum(np.square(self.box_pos - self.goal_pos)))

    def dist_xy(self, pos):
        ''' Return the distance from the robot to an XY position '''
        pos = np.asarray(pos)
        if pos.shape == (3,):
            pos = pos[:2]
        robot_pos = self.world.robot_pos()
        return np.sqrt(np.sum(np.square(pos - robot_pos[:2])))
    
    def dist_xyz(self, pos):
        ''' Return the distance from the robot to an XYZ position '''
        pos = np.asarray(pos)
        assert pos.shape == (3,)
        if 'arm' in self.robot_base:
            link = []
            link_r = []
            for i in range(self.arm_link_n):
                link.append(self.world.body_pos('link_'+ str(i + 1)))
                link_r.append(self.world.body_size('link_'+ str(i + 1))[0])
                assert link[i].shape == (3,)

            min_dist = float('inf')

            for i in range(self.arm_link_n - 1):
                cur_dist, _ = distLinSeg(link[i], link[i + 1], pos, pos)
                cur_dist -= link_r[i]
                min_dist = min(cur_dist, min_dist)
            return min_dist    
        
        robot_pos = self.world.robot_pos()
        return np.sqrt(np.sum(np.square(pos - robot_pos)))

    def world_xy(self, pos):
        ''' Return the world XY vector to a position from the robot '''
        assert pos.shape == (2,)
        return pos - self.world.robot_pos()[:2]

    def ego_xy(self, pos):
        ''' Return the egocentric XY vector to a position from the robot '''
        assert pos.shape == (2,), f'Bad pos {pos}'
        robot_3vec = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        world_3vec = pos_3vec - robot_3vec
        return np.matmul(world_3vec, robot_mat)[:2]  # only take XY coordinates
    
    def ego_xyz(self, pos):
        ''' Return the egocentric XY vector to a position from the robot '''
        assert pos.shape == (3,), f'Bad 3Dpos {pos}'
        robot_3vec = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        world_3vec = pos - robot_3vec
        return np.matmul(world_3vec, robot_mat) # only take XY coordinates
    
    def ego_body_xyz(self, body, pos):
        ''' Return the egocentric XY vector to a position from the robot '''
        assert pos.shape == (3,), f'Bad 3Dpos {pos}'
        body_3vec = self.world.body_pos(body)
        body_mat = self.world.body_mat(body)
        world_3vec = pos - body_3vec
        return np.matmul(world_3vec, body_mat) # only take XY coordinates

    def obs_compass(self, pos):
        '''
        Return a robot-centric compass observation of a list of positions.

        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.

        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        '''
        pos = np.asarray(pos)
        vec_all = np.zeros((len(self.lidar_body), self.compass_shape))
        for body_idx in range(len(self.lidar_body)):
            body = self.lidar_body[body_idx]
            if pos.shape == (2,):
                pos = np.concatenate([pos, [0]])  # Acamdd a zero z-coordinate
            # Get ego vector in world frame
            vec = pos - self.world.body_pos(body)
            # Rotate into frame
            vec = np.matmul(vec, self.world.body_mat(body))
            # Truncate
            vec = vec[:self.compass_shape]
            # Normalize
            vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
            assert vec.shape == (self.compass_shape,), f'Bad vec {vec}'
            vec_all[body_idx,:] = vec
        return vec_all

    def obs_vision(self):
        ''' Return pixels from the robot camera '''
        # Get a render context so we can
        rows, cols = self.vision_size
        width, height = cols, rows
        vision = self.sim.render(width, height, camera_name='vision', mode='offscreen')
        return np.array(vision, dtype='float32') / 255

    def obs_lidar(self, positions, group):
        '''
        Calculate and return a lidar observation.  See sub methods for implementation.
        '''
        if self.lidar_type == 'pseudo':
            return self.obs_lidar_pseudo(positions)
        elif self.lidar_type == 'natural':
            return self.obs_lidar_natural(group)
        else:
            raise ValueError(f'Invalid lidar_type {self.lidar_type}')
    
    def obs_lidar3D(self, positions, group):
        '''
        Calculate and return a lidar observation.  See sub methods for implementation.
        '''
        if self.lidar_type == 'pseudo':
            return self.obs_lidar_pseudo3D(positions)
        # elif self.lidar_type == 'natural':
        #     #TODO self.obs_lidar_natural3D
        #     return self.obs_lidar_natural(group)
        else:
            raise ValueError(f'Invalid lidar_type {self.lidar_type}')

    def obs_lidar_natural(self, group):
        '''
        Natural lidar casts rays based on the ego-frame of the robot.
        Rays are circularly projected from the robot body origin
        around the robot z axis.
        '''
        body = self.model.body_name2id('robot')
        grp = np.asarray([i == group for i in range(int(const.NGROUP))], dtype='uint8')
        pos = np.asarray(self.world.robot_pos(), dtype='float64')
        mat_t = self.world.robot_mat()
        obs = np.zeros(self.lidar_num_bins)
        for i in range(self.lidar_num_bins):
            theta = (i / self.lidar_num_bins) * np.pi * 2
            vec = np.matmul(mat_t, theta2vec(theta))  # Rotate from ego to world frame
            vec = np.asarray(vec, dtype='float64')
            dist, _ = self.sim.ray_fast_group(pos, vec, grp, 1, body)
            if dist >= 0:
                obs[i] = np.exp(-dist)
        return obs

    def obs_lidar_pseudo(self, positions):
        '''
        Return a robot-centric lidar observation of a list of positions.

        Lidar is a set of bins around the robot (divided evenly in a circle).
        The detection directions are exclusive and exhaustive for a full 360 view.
        Each bin reads 0 if there are no objects in that direction.
        If there are multiple objects, the distance to the closest one is used.
        Otherwise the bin reads the fraction of the distance towards the robot.

        E.g. if the object is 90% of lidar_max_dist away, the bin will read 0.1,
        and if the object is 10% of lidar_max_dist away, the bin will read 0.9.
        (The reading can be thought of as "closeness" or inverse distance)

        This encoding has some desirable properties:
            - bins read 0 when empty
            - bins smoothly increase as objects get close
            - maximum reading is 1.0 (where the object overlaps the robot)
            - close objects occlude far objects
            - constant size observation with variable numbers of objects
        '''
        obs = np.zeros(self.lidar_num_bins)
        for pos in positions:
            pos = np.asarray(pos)
            if pos.shape == (3,):
                pos = pos[:2]  # Truncate Z coordinate
            z = complex(*self.ego_xy(pos))  # X, Y as real, imaginary components
            dist = np.abs(z)
            angle = np.angle(z) % (np.pi * 2)
            bin_size = (np.pi * 2) / self.lidar_num_bins
            bin = int(angle / bin_size)
            bin_angle = bin_size * bin
            if self.lidar_max_dist is None:
                sensor = np.exp(-self.lidar_exp_gain * dist)
            else:
                sensor = max(0, self.lidar_max_dist - dist) / self.lidar_max_dist
            obs[bin] = max(obs[bin], sensor)
            # Aliasing
            if self.lidar_alias:
                alias = (angle - bin_angle) / bin_size
                assert 0 <= alias <= 1, f'bad alias {alias}, dist {dist}, angle {angle}, bin {bin}'
                bin_plus = (bin + 1) % self.lidar_num_bins
                bin_minus = (bin - 1) % self.lidar_num_bins
                obs[bin_plus] = max(obs[bin_plus], alias * sensor)
                obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
        return obs

    def obs_lidar_pseudo3D(self, positions):
        '''
        Return a robot-centric lidar observation of a list of positions.

        Lidar is a set of bins around the robot (divided evenly in a circle).
        The detection directions are exclusive and exhaustive for a full 360 view.
        Each bin reads 0 if there are no objects in that direction.
        If there are multiple objects, the distance to the closest one is used.
        Otherwise the bin reads the fraction of the distance towards the robot.

        E.g. if the object is 90% of lidar_max_dist away, the bin will read 0.1,
        and if the object is 10% of lidar_max_dist away, the bin will read 0.9.
        (The reading can be thought of as "closeness" or inverse distance)

        This encoding has some desirable properties:
            - bins read 0 when empty
            - bins smoothly increase as objects get close
            - maximum reading is 1.0 (where the object overlaps the robot)
            - close objects occlude far objects
            - constant size observation with variable numbers of objects
        '''
        obs = np.zeros((len(self.lidar_body), self.lidar_num_bins, self.lidar_num_bins3D))
        for body_idx in range(len(self.lidar_body)):
            body = self.lidar_body[body_idx]
            for pos in positions:
                pos = np.asarray(pos)
                assert pos.shape == (3,)
                x, y, z = self.ego_body_xyz(body, pos)
                proj_xy= complex(x,y)  # X, Y as real, imaginary components
                dist_xy = np.abs(proj_xy)
                angle_xy = np.angle(proj_xy) % (np.pi * 2)

                bin_size_xy = (np.pi * 2) / self.lidar_num_bins
                bin_xy = int(angle_xy / bin_size_xy)
                bin_angle_xy = bin_size_xy * bin_xy

                proj_z = complex(dist_xy, z)
                dist_z = np.abs(proj_z)
                angle_z = np.angle(proj_z) + np.pi / 2
        
                bin_size_z = (np.pi) / self.lidar_num_bins3D
                bin_z = int(angle_z / bin_size_z)
                bin_angle_z = bin_size_z * bin_z

                if self.lidar_max_dist is None:
                    sensor = np.exp(-self.lidar_exp_gain * dist_z)
                else:
                    sensor = max(0, self.lidar_max_dist - dist_z) / self.lidar_max_dist
                obs[body_idx][bin_xy][bin_z] = max(obs[body_idx][bin_xy][bin_z], sensor)
                # Aliasing
                if self.lidar_alias:
                    alias_xy = (angle_xy - bin_angle_xy) / bin_size_xy
                    assert 0 <= alias_xy <= 1, f'bad alias {alias_xy}, dist {dist_xy}, angle {angle_xy}, bin {bin_xy}'
                    bin_plus_xy = (bin_xy + 1) % self.lidar_num_bins
                    bin_minus_xy = (bin_xy - 1) % self.lidar_num_bins
                    obs[body_idx][bin_plus_xy][bin_z]  = max(obs[body_idx][bin_plus_xy][bin_z] , alias_xy * sensor)
                    obs[body_idx][bin_minus_xy][bin_z] = max(obs[body_idx][bin_minus_xy][bin_z] , (1 - alias_xy) * sensor)

                    alias_z = (angle_z - bin_angle_z) / bin_size_z
                    assert 0 <= alias_z <= 1, f'bad alias {alias_z}, dist {dist_z}, angle {angle_z}, bin {bin_z}'
                    bin_plus_z = min((bin_z + 1), self.lidar_num_bins3D - 1)
                    bin_minus_z = max((bin_z - 1), 0)
                    obs[body_idx][bin_xy][bin_plus_z]  = max(obs[body_idx][bin_xy][bin_plus_z] , alias_z * sensor)
                    obs[body_idx][bin_xy][bin_minus_z] = max(obs[body_idx][bin_xy][bin_minus_z] , (1 - alias_z) * sensor)
        return obs

    def obs(self):
        ''' Return the observation of our agent '''
        self.sim.forward()  # Needed to get sensordata correct
        obs = {}

        if self.observe_goal_dist:
            obs['goal_dist'] = np.array([np.exp(-self.dist_goal())])
        if self.observe_goal_comp:
            obs['goal_compass'] = self.obs_compass(self.goal_pos)
        if self.observe_goal_lidar:
            if self.goal_3D:
                obs['goal_lidar'] = self.obs_lidar3D([self.goal_pos], GROUP_GOAL)
            else:
                obs['goal_lidar'] = self.obs_lidar([self.goal_pos], GROUP_GOAL)
        if self.task == 'push':
            box_pos = self.box_pos
            if self.observe_box_comp:
                obs['box_compass'] = self.obs_compass(box_pos)
            if self.observe_box_lidar:
                obs['box_lidar'] = self.obs_lidar([box_pos], GROUP_BOX)
        if self.task == 'circle' and self.observe_circle:
            obs['circle_lidar'] = self.obs_lidar([self.goal_pos], GROUP_CIRCLE)
        if self.observe_freejoint:
            joint_id = self.model.joint_name2id('robot')
            joint_qposadr = self.model.jnt_qposadr[joint_id]
            assert joint_qposadr == 0  # Needs to be the first entry in qpos
            obs['freejoint'] = self.data.qpos[:7]
        if self.observe_com:
            obs['com'] = self.world.robot_com()
        if self.observe_sensors:
            # Sensors which can be read directly, without processing
            for sensor in self.sensors_obs:  # Explicitly listed sensors
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.hinge_vel_names:
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.ballangvel_names:
                obs[sensor] = self.world.get_sensor(sensor)
            # Process angular position sensors
            if self.sensors_angle_components:
                for sensor in self.robot.hinge_pos_names:
                    theta = float(self.world.get_sensor(sensor))  # Ensure not 1D, 1-element array
                    obs[sensor] = np.array([np.sin(theta), np.cos(theta)])
                for sensor in self.robot.ballquat_names:
                    quat = self.world.get_sensor(sensor)
                    obs[sensor] = quat2mat(quat)
            else:  # Otherwise read sensors directly
                for sensor in self.robot.hinge_pos_names:
                    obs[sensor] = self.world.get_sensor(sensor)
                for sensor in self.robot.ballquat_names:
                    obs[sensor] = self.world.get_sensor(sensor)
        if self.observe_remaining:
            obs['remaining'] = np.array([self.steps / self.num_steps])
            assert 0.0 <= obs['remaining'][0] <= 1.0, 'bad remaining {}'.format(obs['remaining'])
        if self.walls_num and self.observe_walls:
            obs['walls_lidar'] = self.obs_lidar(self.walls_pos, GROUP_WALL)
        if self.observe_hazards:
            obs['hazards_lidar'] = self.obs_lidar(self.hazards_pos, GROUP_HAZARD)
        if self.observe_hazard3Ds:
            obs['hazard3Ds_lidar'] = self.obs_lidar3D(self.hazard3Ds_pos, GROUP_HAZARD3D)
        if self.observe_vases:
            obs['vases_lidar'] = self.obs_lidar(self.vases_pos, GROUP_VASE)
        if self.gremlins_num and self.observe_gremlins:
            obs['gremlins_lidar'] = self.obs_lidar(self.gremlins_obj_pos, GROUP_GREMLIN)
        if self.ghosts_num and self.observe_ghosts:
            obs['ghosts_lidar'] = self.obs_lidar(self.ghosts_pos, GROUP_GHOST)
        if self.ghost3Ds_num and self.observe_ghost3Ds:
            obs['ghost3Ds_lidar'] = self.obs_lidar3D(self.ghost3Ds_pos, GROUP_GHOST3D)
        if self.robbers_num and self.observe_robbers:
            obs['robbers_lidar'] = self.obs_lidar(self.robbers_pos, GROUP_ROBBER)
        if self.robber3Ds_num and self.observe_robber3Ds:
            obs['robber3Ds_lidar'] = self.obs_lidar3D(self.robber3Ds_pos, GROUP_ROBBER3D)
        if self.pillars_num and self.observe_pillars:
            obs['pillars_lidar'] = self.obs_lidar(self.pillars_pos, GROUP_PILLAR)
        if self.buttons_num and self.observe_buttons:
            # Buttons observation is zero while buttons are resetting
            if self.buttons_timer == 0:
                obs['buttons_lidar'] = self.obs_lidar(self.buttons_pos, GROUP_BUTTON)
            else:
                obs['buttons_lidar'] = np.zeros(self.lidar_num_bins)
        if self.observe_qpos:
            obs['qpos'] = self.data.qpos.copy()
        if self.observe_qvel:
            obs['qvel'] = self.data.qvel.copy()
        if self.observe_ctrl:
            obs['ctrl'] = self.data.ctrl.copy()
        if self.observe_vision:
            obs['vision'] = self.obs_vision()
        if self.observation_flatten:
            flat_obs = np.zeros(self.obs_flat_size)
            offset = 0
            for k in sorted(self.obs_space_dict.keys()):
                k_size = np.prod(obs[k].shape)
                flat_obs[offset:offset + k_size] = obs[k].flat
                offset += k_size
            obs = flat_obs
        assert self.observation_space.contains(obs), f'Bad obs {obs} {self.observation_space}'
        return obs


    def cost(self):
        ''' Calculate the current costs and return a dict '''
        self.sim.forward()  # Ensure positions and contacts are correct
        cost = {}
        # Conctacts processing
        if self.constrain_vases:
            cost['cost_vases_contact'] = 0
        if self.constrain_pillars:
            cost['cost_pillars'] = 0
        if self.constrain_buttons:
            cost['cost_buttons'] = 0
        if self.constrain_gremlins:
            cost['cost_gremlins'] = 0
        if self.constrain_ghosts:
            cost['cost_ghosts'] = 0
        if self.constrain_ghost3Ds:
            cost['cost_ghost3Ds'] = 0
        buttons_constraints_active = self.constrain_buttons and (self.buttons_timer == 0)
        for contact in self.data.contact[:self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom_id2name(g) for g in geom_ids])
            if self.constrain_vases and any(n.startswith('vase') for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    cost['cost_vases_contact'] += self.vases_contact_cost
            if self.constrain_pillars and any(n.startswith('pillar') for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    cost['cost_pillars'] += self.pillars_cost
            if buttons_constraints_active and any(n.startswith('button') for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    if not any(n == f'button{self.goal_button}' for n in geom_names):
                        cost['cost_buttons'] += self.buttons_cost
            if self.constrain_gremlins and any(n.startswith('gremlin') for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    cost['cost_gremlins'] += self.gremlins_contact_cost
            if self.constrain_ghosts and self.ghosts_contact and any(n.startswith('ghost') for n in geom_names):
                if any(n.startswith('ghost3D') for n in geom_names):
                    continue
                if any(n in self.robot.geom_names for n in geom_names):
                    cost['cost_ghosts'] += self.ghosts_contact_cost
            if self.constrain_ghost3Ds and self.ghost3Ds_contact and any(n.startswith('ghost3D') for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    cost['cost_ghost3Ds'] += self.ghost3Ds_contact_cost

        # Displacement processing
        if self.constrain_vases and self.vases_displace_cost:
            cost['cost_vases_displace'] = 0
            for i in range(self.vases_num):
                name = f'vase{i}'
                dist = np.sqrt(np.sum(np.square(self.data.get_body_xpos(name)[:2] - self.reset_layout[name])))
                if dist > self.vases_displace_threshold:
                    cost['cost_vases_displace'] += dist * self.vases_displace_cost

        # Velocity processing
        if self.constrain_vases and self.vases_velocity_cost:
            # TODO: penalize rotational velocity too, but requires another cost coefficient
            cost['cost_vases_velocity'] = 0
            for i in range(self.vases_num):
                name = f'vase{i}'
                vel = np.sqrt(np.sum(np.square(self.data.get_body_xvelp(name))))
                if vel >= self.vases_velocity_threshold:
                    cost['cost_vases_velocity'] += vel * self.vases_velocity_cost

        # Calculate constraint violations
        if self.constrain_hazards:
            cost['cost_hazards'] = 0
            for h_pos in self.hazards_pos:
                h_dist = self.dist_xy(h_pos)
                if h_dist <= self.hazards_size:
                    cost['cost_hazards'] += self.hazards_cost * (self.hazards_size - h_dist)
        
        if self.constrain_hazard3Ds:
            cost['cost_hazard3Ds'] = 0
            for h_pos in self.hazard3Ds_pos:
                h_dist = self.dist_xyz(h_pos)
                if h_dist <= self.hazard3Ds_size:
                    cost['cost_hazard3Ds'] += self.hazard3Ds_cost * (self.hazard3Ds_size - h_dist)

        # Calculate non-contact cost of ghosts
        if self.constrain_ghosts and (self.ghosts_contact == False):
            for h_pos in self.ghosts_pos:
                h_dist = self.dist_xy(h_pos)
                if h_dist <= self.ghosts_size:
                    cost['cost_ghosts'] += self.ghosts_dist_cost * (self.ghosts_size - h_dist)

        if self.constrain_ghost3Ds and (self.ghost3Ds_contact == False):
            for h_pos in self.ghost3Ds_pos:
                h_dist = self.dist_xyz(h_pos)
                if h_dist <= self.ghost3Ds_size:
                    cost['cost_ghost3Ds'] += self.ghost3Ds_dist_cost * (self.ghost3Ds_size - h_dist)

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))

        # Optionally remove shaping from reward functions.
        if self.constrain_indicator:
            for k in list(cost.keys()):
                cost[k] = float(cost[k] > 0.0)  # Indicator function

        self._cost = cost

        return cost

    def goal_met(self):
        ''' Return true if the current goal is met this step '''
        if self.task == 'goal':
            return self.dist_goal() <= self.goal_size
        if self.task == 'push':
            return self.dist_box_goal() <= self.goal_size
        if self.task == 'button':
            for contact in self.data.contact[:self.data.ncon]:
                geom_ids = [contact.geom1, contact.geom2]
                geom_names = sorted([self.model.geom_id2name(g) for g in geom_ids])
                if any(n == f'button{self.goal_button}' for n in geom_names):
                    if any(n in self.robot.geom_names for n in geom_names):
                        return True
            return False
        if self.task in ['x', 'z', 'circle', 'none','chase', 'defense']:
            return False
        raise ValueError(f'Invalid task {self.task}')

    def set_mocaps(self):
        ''' Set mocap object positions before a physics step is executed '''
        if self.gremlins_num: # self.constrain_gremlins:
            phase = float(self.data.time)
            for i in range(self.gremlins_num):
                name = f'gremlin{i}'
                target = np.array([np.sin(phase), np.cos(phase)]) * self.gremlins_travel
                pos = np.r_[target, [self.gremlins_size]]
                self.data.set_mocap_pos(name + 'mocap', pos)
        if 'arm' in self.robot_base:
            robot_pos = self.arm_end_pos
        else:
            robot_pos = self.world.robot_pos()
        if self.ghosts_num:
            phase = float(self.data.time)
            ghost_pos_last_dict = []
            for i in range(self.ghosts_num):
                name = f'ghost{i}'
                ghost_origin = self.layout[name]
                ghost_pos_mocap = self.data.get_body_xpos(name +'mocap').copy()
                ghost_pos_last_dict.append(ghost_pos_mocap[:2] + ghost_origin)
            for i in range(self.ghosts_num):
                name = f'ghost{i}'
                ghost_origin = self.layout[name]
                ghost_pos_mocap = self.data.get_body_xpos(name +'mocap').copy()
                ghost_pos_last = ghost_pos_mocap[:2] + ghost_origin
                target =  ghost_pos_mocap[:2]
                for j in range(self.ghosts_num):
                    dist_ij = ghost_pos_last_dict[j] - ghost_pos_last_dict[i]
                    if j != i and np.sqrt(np.sum(np.square(dist_ij))) < 2 * self.ghosts_size:
                        direction = - dist_ij / np.sqrt(np.sum(np.square(dist_ij)))
                        target += self.ghosts_velocity*direction
                if np.sqrt(np.sum(np.square(ghost_pos_last))) > self.ghosts_travel:
                    direction = - ghost_pos_last / np.sqrt(np.sum(np.square(ghost_pos_last)))
                    target += self.ghosts_velocity*direction
                else:
                    direction = robot_pos[:2] - ghost_pos_last
                    norm = np.sqrt(np.sum(np.square(direction)))
                    direction_norm = direction / norm
                    if norm < self.ghosts_safe_dist:  
                        target = ghost_pos_mocap[:2]
                    else:
                        target = ghost_pos_mocap[:2] + self.ghosts_velocity*direction_norm
                
                pos = np.r_[target, 2e-2]
                self.data.set_mocap_pos(name + 'mocap', pos)
        if self.ghost3Ds_num:
            ghost3D_pos_last_dict = []
            for i in range(self.ghost3Ds_num):
                name = f'ghost3D{i}'
                ghost_origin = np.r_[self.layout[name], self._ghost3Ds_z[i]]
                ghost_pos_mocap = self.data.get_body_xpos(name +'mocap').copy()
                ghost3D_pos_last_dict.append(ghost_pos_mocap + ghost_origin)
            for i in range(self.ghost3Ds_num):
                name = f'ghost3D{i}'
                ghost_origin = np.r_[self.layout[name], self._ghost3Ds_z[i]]
                ghost_pos_mocap = self.data.get_body_xpos(name +'mocap').copy()
                ghost_pos_last = ghost_pos_mocap + ghost_origin
                target =  ghost_pos_mocap
                for j in range(self.ghost3Ds_num):
                    dist_ij = ghost3D_pos_last_dict[j] - ghost3D_pos_last_dict[i]
                    if j != i and np.sqrt(np.sum(np.square(dist_ij))) < 2 * self.ghost3Ds_size:
                        direction = - dist_ij / np.sqrt(np.sum(np.square(dist_ij)))
                        target += self.ghost3Ds_velocity*direction
                if np.sqrt(np.sum(np.square(ghost_pos_last[:2]))) > self.ghost3Ds_travel:
                    direction = - ghost_pos_last[:2] / np.sqrt(np.sum(np.square(ghost_pos_last[:2])))
                    target += np.r_[self.ghost3Ds_velocity*direction, 0]
                else:
                    direction = robot_pos - ghost_pos_last
                    norm = np.sqrt(np.sum(np.square(direction)))
                    direction_norm = direction / norm
                    if norm < self.ghost3Ds_safe_dist: 
                        target = ghost_pos_mocap
                    else:
                        target = ghost_pos_mocap + self.ghost3Ds_velocity*direction_norm
                        
                target[2] = np.clip(target[2], self.ghost3Ds_z_range[0], self.ghost3Ds_z_range[1])              
                pos = target
                self.data.set_mocap_pos(name + 'mocap', pos)
        if self.robbers_num: 
            robber_pos_last_dict = []
            for i in range(self.robbers_num):
                name = f'robber{i}'
                robber_origin = self.layout[name]
                robber_pos_mocap = self.data.get_body_xpos(name +'mocap').copy()
                robber_pos_last_dict.append(robber_pos_mocap[:2] + robber_origin)
            for i in range(self.robbers_num):
                name = f'robber{i}'
                robber_origin = self.layout[name]
                robber_pos_mocap = self.data.get_body_xpos(name +'mocap').copy()
                robber_pos_last = robber_pos_mocap[:2] + robber_origin
                target =  robber_pos_mocap[:2]
                for j in range(self.robbers_num):
                    dist_ij = robber_pos_last_dict[j] - robber_pos_last_dict[i]
                    if j != i and np.sqrt(np.sum(np.square(dist_ij))) < 2 * self.robbers_size:
                        direction = - dist_ij / np.sqrt(np.sum(np.square(dist_ij)))
                        target += self.robbers_velocity*direction
                if np.sqrt(np.sum(np.square(robber_pos_last))) > self.robbers_travel:
                    direction = - robber_pos_last / np.sqrt(np.sum(np.square(robber_pos_last)))
                    target += self.robbers_velocity*direction
                else:
                    direction = robot_pos[:2] - robber_pos_last
                    norm = np.sqrt(np.sum(np.square(direction)))
                    direction_norm = direction / norm
                    if self.task == 'defense':
                        if norm > self.defense_range:
                            direction_norm = - robber_pos_last[:2] / np.sqrt(np.sum(np.square(robber_pos_last[:2])))
                            target = robber_pos_mocap[:2] + self.robbers_velocity*direction_norm
                        else:
                            target = robber_pos_mocap[:2] - self.robbers_velocity*10*direction_norm
                    elif self.task == 'chase':
                        if norm > self.chase_range:
                            target = robber_pos_mocap[:2]
                        else:
                            target = robber_pos_mocap[:2] - self.robbers_velocity*10*direction_norm
                    else:
                        target = robber_pos_mocap[:2] - self.robbers_velocity*direction_norm
                
                pos = np.r_[target, 2e-2]
                self.data.set_mocap_pos(name + 'mocap', pos)
        if self.robber3Ds_num: # self.constrain_gremlins:
            phase = float(self.data.time)
            robber3D_pos_last_dict = []
            for i in range(self.robber3Ds_num):
                name = f'robber3D{i}'
                robber_origin = np.r_[self.layout[name], self._robber3Ds_z[i]]
                robber_pos_mocap = self.data.get_body_xpos(name +'mocap').copy()
                robber3D_pos_last_dict.append(robber_pos_mocap + robber_origin)
            for i in range(self.robber3Ds_num):
                name = f'robber3D{i}'
                robber_origin = np.r_[self.layout[name], self._robber3Ds_z[i]]
                robber_pos_mocap = self.data.get_body_xpos(name +'mocap').copy()
                robber_pos_last = robber_pos_mocap + robber_origin
                target =  robber_pos_mocap
                for j in range(self.robber3Ds_num):
                    dist_ij = robber3D_pos_last_dict[j] - robber3D_pos_last_dict[i]
                    if j != i and np.sqrt(np.sum(np.square(dist_ij))) < 2 * self.robber3Ds_size:
                        direction = - dist_ij / np.sqrt(np.sum(np.square(dist_ij)))
                        target += self.robber3Ds_velocity*direction
                if np.sqrt(np.sum(np.square(robber_pos_last[:2]))) > self.robber3Ds_travel:
                    direction = - robber_pos_last[:2] / np.sqrt(np.sum(np.square(robber_pos_last[:2])))
                    target += np.r_[self.robber3Ds_velocity*direction, 0]
                else:
                    direction = robot_pos - robber_pos_last
                    norm = np.sqrt(np.sum(np.square(direction)))
                    direction_norm = direction / norm
                    if self.task == 'defense':
                        if norm > self.defense_range:
                            direction_norm = - robber_pos_last[:2] / np.sqrt(np.sum(np.square(robber_pos_last[:2])))
                            target = robber_pos_mocap + self.robber3Ds_velocity*np.r_[direction_norm,0]
                        else:
                            target = robber_pos_mocap - self.robber3Ds_velocity*10*direction_norm
                    elif self.task == 'chase':
                        if norm > self.chase_range:
                            target = robber_pos_mocap
                        else:
                            target = robber_pos_mocap - self.robber3Ds_velocity*10*direction_norm
                    else:
                        target = robber_pos_mocap - self.robbers_velocity*direction_norm
                        
                target[2] = np.clip(target[2], self.robber3Ds_z_range[0], self.robber3Ds_z_range[1])              
                pos = target
                self.data.set_mocap_pos(name + 'mocap', pos)
    def update_layout(self):
        ''' Update layout dictionary with new places of objects '''
        self.sim.forward()
        for k in list(self.layout.keys()):
            # Mocap objects have to be handled separately
            if 'gremlin' in k or 'ghost' in k or 'robber' in k:
                continue
            self.layout[k] = self.data.get_body_xpos(k)[:2].copy()

    def buttons_timer_tick(self):
        ''' Tick the buttons resampling timer '''
        self.buttons_timer = max(0, self.buttons_timer - 1)

    def step(self, action):
        ''' Take a step and return observation, reward, done, and info '''
        action = np.array(action, copy=False)  # Cast to ndarray
        assert not self.done, 'Environment must be reset before stepping'

        info = {}

        # Set action
        if "drone" in self.robot_base:
            action = np.clip(action, -1.0, 1.0)
            mass = self.model.body_mass[self.model.body_name2id('robot')]
            mass += self.model.body_mass[self.model.body_name2id('p1')]
            mass += self.model.body_mass[self.model.body_name2id('p2')]
            mass += self.model.body_mass[self.model.body_name2id('p3')]
            mass += self.model.body_mass[self.model.body_name2id('p4')]
            robot_pos = self.world.robot_pos()
            R = self.world.robot_mat()
            f = mass * 9.81
            if (robot_pos[2] > 3):
                f = 0

            torque = [0, 0, 0]
            for i in range(4):
                propeller = 'p' + str(i + 1)
                force = [0, 0, f/4 + action[i]*1e-1]
                force = np.array(force).reshape(3,1)
                force = R@force
                force = [force[i,0] for i in range(3)]
                self.data.xfrc_applied[self.model.body_name2id(propeller),:] = force + torque
        else:
            action_range = self.model.actuator_ctrlrange
            # action_scale = action_range[:,1] - action_range[:, 0]
            self.data.ctrl[:] = np.clip(action, action_range[:,0], action_range[:,1]) #np.clip(action * 2 / action_scale, -1, 1)
            if self.action_noise:
                self.data.ctrl[:] += self.action_noise * self.rs.randn(self.model.nu)

        # Simulate physics forward
        exception = False
        for _ in range(self.rs.binomial(self.frameskip_binom_n, self.frameskip_binom_p)):
            try:
                self.set_mocaps()
                self.sim.step()  # Physics simulation step
            except MujocoException as me:
                print('MujocoException', me)
                exception = True
                break
        if exception:
            self.done = True
            reward = self.reward_exception
            info['cost_exception'] = 1.0
        else:
            self.sim.forward()  # Needed to get sensor readings correct!

            # Reward processing
            reward = self.reward()

            # Constraint violations
            info.update(self.cost())

            # Button timer (used to delay button resampling)
            self.buttons_timer_tick()

            # Goal processing
            if self.goal_met():
                info['goal_met'] = True
                reward += self.reward_goal
                if self.continue_goal:
                    # Update the internal layout so we can correctly resample (given objects have moved)
                    self.update_layout()
                    # Reset the button timer (only used for task='button' environments)
                    self.buttons_timer = self.buttons_resampling_delay
                    # Try to build a new goal, end if we fail
                    if self.terminate_resample_failure:
                        try:
                            self.build_goal()
                        except ResamplingError as e:
                            # Normal end of episode
                            self.done = True
                    else:
                        # Try to make a goal, which could raise a ResamplingError exception
                        self.build_goal()
                else:
                    self.done = True

        # Timeout
        self.steps += 1
        if self.steps >= self.num_steps:
            self.done = True  # Maximum number of steps in an episode reached

        return self.obs(), reward, self.done, info

    def reward(self):
        ''' Calculate the dense component of reward.  Call exactly once per step '''
        reward = 0.0
        # Distance from robot to goal
        if self.task in ['goal', 'button']:
            dist_goal = self.dist_goal()
            reward += (self.last_dist_goal - dist_goal) * self.reward_distance
            self.last_dist_goal = dist_goal
        # Distance from robot to box
        if self.task == 'push':
            dist_box = self.dist_box()
            gate_dist_box_reward = (self.last_dist_box > self.box_null_dist * self.box_size)
            reward += (self.last_dist_box - dist_box) * self.reward_box_dist * gate_dist_box_reward
            self.last_dist_box = dist_box
        # Distance from box to goal
        if self.task == 'push':
            dist_box_goal = self.dist_box_goal()
            reward += (self.last_box_goal - dist_box_goal) * self.reward_box_goal
            self.last_box_goal = dist_box_goal
        # Used for forward locomotion tests
        if self.task == 'x':
            robot_com = self.world.robot_com()
            reward += (robot_com[0] - self.last_robot_com[0]) * self.reward_x
            self.last_robot_com = robot_com
        # Used for jump up tests
        if self.task == 'z':
            robot_com = self.world.robot_com()
            reward += (robot_com[2] - self.last_robot_com[2]) * self.reward_z
            self.last_robot_com = robot_com
        # Circle environment reward
        if self.task == 'circle':
            robot_com = self.world.robot_com()
            robot_vel = self.world.robot_vel()
            x, y, _ = robot_com
            u, v, _ = robot_vel
            radius = np.sqrt(x**2 + y**2)
            reward += (((-u*y + v*x)/radius)/(1 + np.abs(radius - self.circle_radius))) * self.reward_circle
        if self.task == 'chase':
            dist_robber = 1000
            for h_pos in self.robbers_pos:
                dist_robber = min(dist_robber, self.dist_xyz(h_pos))
            for h_pos in self.robber3Ds_pos:
                dist_robber = min(dist_robber, self.dist_xyz(h_pos))
            if self.last_dist_robber != -1:
                reward += (self.last_dist_robber - dist_robber) * self.reward_chase
            self.last_dist_robber = dist_robber
        if self.task == 'defense':
            for h_pos in self.robbers_pos:
                dist = np.sqrt(np.sum(np.square(h_pos[:2]))) / self.circle_radius - 1
                if dist < 0:
                    reward += dist * self.reward_defense
                else:
                    reward += 0.1 * self.reward_defense
            for h_pos in self.robber3Ds_pos:
                dist = np.sqrt(np.sum(np.square(h_pos[:2]))) / self.circle_radius - 1
                if dist < 0:
                    reward += dist * self.reward_defense
                else:
                    reward += 0.1 * self.reward_defense
        # Intrinsic reward for uprightness
        if self.reward_orientation:
            zalign = quat2zalign(self.data.get_body_xquat(self.reward_orientation_body))
            reward += self.reward_orientation_scale * zalign
        # Clip reward
        if self.reward_clip:
            in_range = reward < self.reward_clip and reward > -self.reward_clip
            if not(in_range):
                reward = np.clip(reward, -self.reward_clip, self.reward_clip)
                print('Warning: reward was outside of range!')
        return reward

    def render_lidar(self, poses, color, offset, group):
        ''' Render the lidar observation '''
        robot_pos = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        lidar = self.obs_lidar(poses, group)
        for i, sensor in enumerate(lidar):
            if self.lidar_type == 'pseudo':
                i += 0.5  # Offset to center of bin
            theta = 2 * np.pi * i / self.lidar_num_bins
            rad = self.render_lidar_radius
            
            binpos = np.array([np.cos(theta) * rad, np.sin(theta) * rad, offset])
            pos = robot_pos + np.matmul(binpos, robot_mat.transpose())
            alpha = min(1, sensor + .1)
            self.viewer.add_marker(pos=pos,
                                   size=self.render_lidar_size * np.ones(3),
                                   type=const.GEOM_SPHERE,
                                   rgba=np.array(color) * alpha,
                                   label='')
    
    def render_lidar3D(self, poses, color, offset, group):
        ''' Render the lidar observation '''
        
        lidar = self.obs_lidar3D(poses, group)
        rad = self.render_lidar_radius + offset
        for body_idx in range(len(self.lidar_body)):
            body = self.lidar_body[body_idx]
            body_pos = self.world.body_pos(body)
            body_mat = self.world.body_mat(body)
            for i, row in enumerate(lidar[body_idx]):
                if self.lidar_type == 'pseudo':
                        i += 0.5  # Offset to center of bin
                theta_xy = 2 * np.pi * i / self.lidar_num_bins
                
                for j, sensor in enumerate(row):
                    if self.lidar_type == 'pseudo':
                        j += 0.5  # Offset to center of bin
                    theta_z = -np.pi / 2 + np.pi * j / self.lidar_num_bins3D
                    h_z = np.sin(theta_z) * rad
                    rad_xy = rad * np.cos(theta_z)
                    binpos = np.array([np.cos(theta_xy) * rad_xy, np.sin(theta_xy) * rad_xy, h_z])
                    pos = body_pos + np.matmul(binpos, body_mat.transpose())
                    alpha = min(1, sensor + .1)
                    self.viewer.add_marker(pos=pos,
                                        size=self.render_lidar_size * np.ones(3),
                                        type=const.GEOM_SPHERE,
                                        rgba=np.array(color) * alpha,
                                        label='')
            


    def render_compass(self, pose, color, offset):
        ''' Render a compass observation '''
        compass = self.obs_compass(pose)
        for body_idx in range(len(self.lidar_body)):
            body = self.lidar_body[body_idx]
            body_pos = self.world.body_pos(body)
            body_mat = self.world.body_mat(body)
            if self.compass_shape == 3:
                compass_pos = compass[body_idx,] * 0.3
            if self.compass_shape == 2:
                compass_pos = np.concatenate([compass[body_idx, :2] * 0.15, [offset]])
            pos = body_pos + np.matmul(compass_pos, body_mat.transpose())
            self.viewer.add_marker(pos=pos,
                                size=.05 * np.ones(3),
                                type=const.GEOM_SPHERE,
                                rgba=np.array(color) * 0.5,
                                label='')

    def render_area(self, pos, size, color, label='', alpha=0.1):
        ''' Render a radial area in the environment '''
        z_size = min(size, 0.3)
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(pos=pos,
                               size=[size, size, z_size],
                               type=const.GEOM_CYLINDER,
                               rgba=np.array(color) * alpha,
                               label=label if self.render_labels else '')

    def render_sphere(self, pos, size, color, label='', alpha=0.1):
        ''' Render a radial area in the environment '''
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(pos=pos,
                               size=size * np.ones(3),
                               type=const.GEOM_SPHERE,
                               rgba=np.array(color) * alpha,
                               label=label if self.render_labels else '')

    def render_swap_callback(self):
        ''' Callback between mujoco render and swapping GL buffers '''
        if self.observe_vision and self.vision_render:
            self.viewer.draw_pixels(self.save_obs_vision, 0, 0)

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 0         # id of the body to track ()
        # self.viewer.cam.distance = self.model.stat.extent * 3       # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.distance = 5
        self.viewer.cam.lookat[0] = 0         # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] = -3
        self.viewer.cam.lookat[2] = 5
        self.viewer.cam.elevation = -60           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 90              # camera rotation around the camera's vertical axis

    def render(self,
               mode='human', 
               camera_id=-1,
               width=DEFAULT_WIDTH,
               height=DEFAULT_HEIGHT
               ):
        ''' Render the environment to the screen '''
        
        if self.viewer is None or mode!=self._old_render_mode:
            # Set camera if specified
            if mode == 'human':

                self.viewer = MjViewer(self.sim)
                self.viewer.cam.fixedcamid = -1
                self.viewer.cam.type = const.CAMERA_FREE
            else:
                self.viewer = MjRenderContextOffscreen(self.sim)
                self.viewer._hide_overlay = True
                self.viewer.cam.fixedcamid = camera_id #self.model.camera_name2id(mode)
                # self.viewer.cam.type = const.CAMERA_FIXED
                self.viewer.cam.type = const.CAMERA_FREE
            self.viewer_setup()
            self.viewer.render_swap_callback = self.render_swap_callback
            # Turn all the geom groups on
            self.viewer.vopt.geomgroup[:] = 1
            self._old_render_mode = mode
        # print('here2', self.viewer.cam.fixedcamid)
        
        self.viewer.update_sim(self.sim)
        self.viewer._hide_overlay = True
        if camera_id is not None:
            # Update camera if desired
            self.viewer.cam.fixedcamid = camera_id

        

        # Lidar markers
        if self.render_lidar_markers:
            offset = self.render_lidar_offset_init  # Height offset for successive lidar indicators
            offset3D = 0.1 # Height offset for successive lidar indicators
            if 'box_lidar' in self.obs_space_dict or 'box_compass' in self.obs_space_dict:
                if 'box_lidar' in self.obs_space_dict:
                    self.render_lidar([self.box_pos], COLOR_BOX, offset, GROUP_BOX)
                if 'box_compass' in self.obs_space_dict:
                    self.render_compass(self.box_pos, COLOR_BOX, offset)
                offset += self.render_lidar_offset_delta
            if 'goal_lidar' in self.obs_space_dict or 'goal_compass' in self.obs_space_dict:
                if 'goal_lidar' in self.obs_space_dict:
                    if self.goal_3D:
                        self.render_lidar3D([self.goal_pos], COLOR_GOAL, offset3D, GROUP_GOAL)
                        offset3D += self.render_lidar_offset_delta
                    else:
                        self.render_lidar([self.goal_pos], COLOR_GOAL, offset, GROUP_GOAL)
                        offset += self.render_lidar_offset_delta
                if 'goal_compass' in self.obs_space_dict:
                    self.render_compass(self.goal_pos, COLOR_GOAL, offset)
            if 'buttons_lidar' in self.obs_space_dict:
                self.render_lidar(self.buttons_pos, COLOR_BUTTON, offset, GROUP_BUTTON)
                offset += self.render_lidar_offset_delta
            if 'circle_lidar' in self.obs_space_dict:
                self.render_lidar([ORIGIN_COORDINATES], COLOR_CIRCLE, offset, GROUP_CIRCLE)
                offset += self.render_lidar_offset_delta
            if 'walls_lidar' in self.obs_space_dict:
                self.render_lidar(self.walls_pos, COLOR_WALL, offset, GROUP_WALL)
                offset += self.render_lidar_offset_delta
            if 'hazards_lidar' in self.obs_space_dict:
                self.render_lidar(self.hazards_pos, COLOR_HAZARD, offset, GROUP_HAZARD)
                offset += self.render_lidar_offset_delta
            if 'hazard3Ds_lidar' in self.obs_space_dict:
                self.render_lidar3D(self.hazard3Ds_pos, COLOR_HAZARD3D, offset3D, GROUP_HAZARD3D)
                offset3D += self.render_lidar_offset_delta
            if 'pillars_lidar' in self.obs_space_dict:
                self.render_lidar(self.pillars_pos, COLOR_PILLAR, offset, GROUP_PILLAR)
                offset += self.render_lidar_offset_delta
            if 'gremlins_lidar' in self.obs_space_dict:
                self.render_lidar(self.gremlins_obj_pos, COLOR_GREMLIN, offset, GROUP_GREMLIN)
                offset += self.render_lidar_offset_delta
            if 'ghosts_lidar' in self.obs_space_dict:
                self.render_lidar(self.ghosts_pos, COLOR_GHOST, offset, GROUP_GHOST)
                offset += self.render_lidar_offset_delta
            if 'ghost3Ds_lidar' in self.obs_space_dict:
                self.render_lidar3D(self.ghost3Ds_pos, COLOR_GHOST3D, offset3D, GROUP_GHOST3D)
                offset3D += self.render_lidar_offset_delta
            if 'robbers_lidar' in self.obs_space_dict:
                self.render_lidar(self.robbers_pos, COLOR_ROBBER, offset, GROUP_ROBBER3D)
                offset += self.render_lidar_offset_delta
            if 'robber3Ds_lidar' in self.obs_space_dict:
                self.render_lidar3D(self.robber3Ds_pos, COLOR_ROBBER3D, offset3D, GROUP_ROBBER3D)
                offset3D += self.render_lidar_offset_delta
            if 'vases_lidar' in self.obs_space_dict:
                self.render_lidar(self.vases_pos, COLOR_VASE, offset, GROUP_VASE)
                offset += self.render_lidar_offset_delta

        # Add goal marker
        if self.task == 'button':
            self.render_area(self.goal_pos, self.buttons_size * 2, COLOR_BUTTON, 'goal', alpha=0.1)

        # Add indicator for nonzero cost
        if self._cost.get('cost', 0) > 0:
            self.render_sphere(self.world.robot_pos(), 0.5, COLOR_RED, alpha=.5)

        # Draw vision pixels
        if self.observe_vision and self.vision_render:
            vision = self.obs_vision()
            vision = np.array(vision * 255, dtype='uint8')
            vision = Image.fromarray(vision).resize(self.vision_render_size)
            vision = np.array(vision, dtype='uint8')
            self.save_obs_vision = vision

        if mode=='human':
            self.viewer.render()
        elif mode=='rgb_array':
            self.viewer.render(width, height)
            data = self.viewer.read_pixels(width, height, depth=False)
            self.viewer._markers[:] = []
            self.viewer._overlay.clear()
            return data[::-1, :, :]