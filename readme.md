# GUARD: **G**eneralized **U**nified S**A**fe **R**einforcement Learning **D**evelopment Benchmark

Paper link: [GUARD: A Safe Reinforcement Learning Benchmark](https://arxiv.org/abs/2305.13681)

GUARD is a highly customizable generalized benchmark with a wide variety of RL agents, tasks, and safety constraint specifications.
GUARD comprehensively covers state-of-the-art safe RL algorithms with self-contained implementations.

GUARD is composed of two main components: GUARD Safe RL library and GUARD testing suite.

Supported algorithms in the GUARD Safe RL library include:

**Unconstrained**
- [Absolute Policy Optimization (APO)](todo) (not in paper)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/2308.05585)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [Asynchronous Actor-critic (A2C)](https://arxiv.org/abs/1602.01783?context=cs.LG)


**End-to-end**
- [Constrained Policy Optimization (CPO)](https://arxiv.org/abs/1705.10528)
- [TRPO-Lagrangian](https://arxiv.org/abs/1902.04623)
- [TRPO-Feasible Actor Critic (FAC)](https://arxiv.org/abs/2105.10682)
- [TRPO-Interior-point Policy Optimization (IPO)](https://arxiv.org/abs/1910.09615)
- [Projection-based Constrained Policy Optimization (PCPO)](https://arxiv.org/abs/2010.03152)
- [Primal-Dual Optimization (PDO)](https://arxiv.org/abs/1512.01629) (not in paper)

**Hierarchical**
- [TRPO-Safety Layer (SL)](https://arxiv.org/abs/1801.08757)
- [TRPO-Unrolling Safety Layer (USL)](https://arxiv.org/abs/2206.08528)
- [Lyapunov-based Safe Policy Optimization (LPG)](https://arxiv.org/abs/1901.10031) (not in paper)

GUARD testing suite supports the following agents:
- Swimmer
- Ant
- Walker
- Humanoid
- Hopper
- Arm3
- Arm6
- Drone

GUARD testing suite supports the following tasks:
- Goal
- Push
- Chase
- Defense

GUARD testing suite supports the following safety constraints (obstacles):
- 3D Hazards
- Ghosts
- 3D Ghosts
- Vases
- Pillars
- Buttons
- Gremlins

Obstacles can be either trespassable/untrespassable, immovable/passively movable/actively movable, and pertained to 2D/3D spaces.
For full options, please see the paper.

---
## Installation

Install [mujoco_py](https://github.com/openai/mujoco-py), see the mujoco_py documentation for details. Note that mujoco_py **requires Python 3.6 or greater**.

Afterwards, simply install `safe_rl_envs` by:

```
cd safe_rl_envs
pip install -e .
```

Install environment:
```
conda create --name venv --file requirements.txt
```
---
## Quick Start
### 1. Environment Configuration
A set of pre-configured environments can be found in  `safe_rl_env_config.py`. Our traning process will automatically create the pre-configured environments with `--task <env name>`. 

For a complete list of pre-configured environments, see below.

To create a custom environment using the GUARD Safe RL engine, update the `safe_rl_env_config.py` with custom configurations. For example, to build an environment with a drone robot, the chase task, two dynamic targets, some 3D ghosts,  with constraints on entering the 3D ghosts areas. Add the following configuration to `safe_rl_env_config.py`:

```
if task == "Custom_Env":
  config = {
              # robot setting
              'robot_base': 'xmls/drone.xml',  

              # task setting
              'task': 'defense',
              'goal_3D': True,
              'goal_z_range': [0.5,1.5],
              'goal_size': 0.5,
              'defense_range': 2.5,

              # observation setting
              'observe_robber3Ds': True,
              'observe_ghost3Ds': True, 
              'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                              'touch_p1a', 'touch_p1b', 'touch_p2a', 'touch_p2b',
                              'touch_p3a', 'touch_p3b', 'touch_p4a', 'touch_p4b'],
              
              # constraint setting
              'constrain_ghost3Ds': True, 
              'constrain_indicator': False, 

              # lidar setting
              'lidar_num_bins': 10,
              'lidar_num_bins3D': 6,
              
              # object setting
              'ghost3Ds_num': 8,
              'ghost3Ds_size': 0.3,
              'ghost3Ds_travel':2.5,
              'ghost3Ds_safe_dist': 1.5,
              'ghost3Ds_z_range': [0.5, 1.5],
              'robber3Ds_num': 2,
              'robber3Ds_size': 0.3,
              'robber3Ds_z_range': [0.5, 1.5],
          }
```
The custom environment can then be  used with `--task Custom_Env` in the training process below.
### 2. Benchmark Suite

An environment in the GUARD Safe RL suite is formed as a combination of a task(one of `Goal`, `Push`, `Chase` or `Defense`), a robot (one of `Point`, `Swimmer`, `Ant`, `Walker`, `Humanoid`, `Hopper`, `Arm3`, `Arm6` or `Drone`), and a type of constraints (one of `8Hazards` and `8Ghosts`, `8` is the number of constraints). Environments include:

* `Goal_{Robot}_8Hazards`: A robot must navigate to a goal while avoiding hazards.
* `Goal_{Robot}_8Ghosts`: A robot must navigate to a goal while avoiding ghosts.
* `Push_{Robot}_8Hazards`: A robot must push a box to a goal while avoiding hazards.
* `Push_{Robot}_8Ghosts`: A robot must push a box to a goal while avoiding ghosts.
* `Chase_{Robot}_8Hazards`: A robot must chase two dynamic targets while avoiding hazards.
* `Chase_{Robot}_8Ghosts`: A robot must chase two dynamic targets while avoiding ghosts.
* `Defense_{Robot}_8Hazards`: A robot must prevent two dynamic targets from entering a protected circle area while avoiding hazards.
* `Defense_{Robot}_8Ghosts`: A robot must prevent two dynamic targets from entering a protected circle area while avoiding ghosts.

(To make one of the above, make sure to substitute `Point`, `Swimmer`, `Ant`, `Walker`, `Humanoid`, `Hopper`, `Arm3`, `Arm6` or `Drone`.)


### 3. Training
Take CPO training for example:
```
cd safe_rl_lib/cpo
conda activate venv
python cpo.py --task Goal_Point_8Hazards --seed 1
```
Training logs (e.g., config, model) will be saved under `<algo>/logs/` (in the above example `cpo/logs/`).

### 4. Viualization
To test a trained RL agent on a task and save the video:
```
python cpo_video.py --model_path logs/<exp name>/<exp name>_s<seed>/pyt_save/model.pt --task <env name> --video_name <video name> --max_epoch <max epoch>            
```
To plot training statistics (e.g., reward, cost), copy the all desired log folders to `comparison/` and then run the plot script as follows:
```
cd safe_rl_lib
mkdir comparison
cp -r <algo>/logs/<exp name> comparison/
python utils/plot.py comparison/ --title <title name> --reward --cost
```

\<title name\> can be anything that describes the current comparison (e.g., "all end-to-end methods").


<!-- ---
## Reproduceing Paper Results
@Weiye -->

---
## Citing GUARD
```
@article{zhao2023guard,
  title={GUARD: A Safe Reinforcement Learning Benchmark},
  author={Zhao, Weiye and Chen, Rui and Sun, Yifan and Liu, Ruixuan and Wei, Tianhao and Liu, Changliu},
  journal={arXiv preprint arXiv:2305.13681},
  year={2023}
}
```

