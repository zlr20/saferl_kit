训练：
python main.py --use_td3 --save_model  --exp_name td3 --seed 2
python main.py --use_epf --save_model  --exp_name epf --seed 2

python train_metadrive.py --use_epo --save_model  --exp_name 100w_epo_seed1500  --seed 1500 
python train_metadrive.py --use_qpsl --save_model --exp_name 100w_qpsl_seed1500 --seed 1500  --delta 0.02 

绘制训练曲线：
./plot.sh


<<<<<<< HEAD
# Saferl kit: Evaluating Efficient Reinforcement Learning Methods for Safe Autonomous Driving



![image-20220517135619231](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20220517135619231.png)

## Description

Safe_RL Kit is a toolkit for developing and evaluating reinforcement learning algorithms, and an accompanying library of state-of-the-art implementations built using that toolkit.

Safe_RL kit 是一个用于基准测试面向自动驾驶任务的高效安全RL方法的工具包，该工具包包含了以下几个部分：

- 几个最先进的(SOTA)安全RL算法的实现：[算法]在本文中，他们被通过离线以及的方式来进行实现，
- 一种新的一阶优化方法EPO：EPO利用单个惩罚因子和ReLU算子来构造一个等价的无约束目标。经验结果表明，该简单的技术对于面向ad的任务具有惊人的有效和性能。
- 

## Repository Structure

The repository is organized as follows：

`/metadrive`车速控制环境和无人驾驶环境分别被封装在Safety-bullet-gym 和 metadrive 文件夹中。

`/bullet_safety_gym` 

`/safe_algo：定义了本文所使用的安全强化学习算法，所有的算法都是基于TD3结构来实现，所有方法的类的结构都是一样的，只是类内方法的实现方式有所不同。

`/saferl_utils`：该文件夹定义了所有算法的经验回放池和网络结构。与`safe_algos`对应，`networks`定义了本文所使用到的网络，所有网络结构均为基于TD3算法的网络结构构建，即2个256的全连接神经网络。`replay_buffer.py`定义了本文算法所涉及到的所有的经验池，其中`RecReplayBuffer`和`SafetyLayerReplayBuffer`是针对[算法paper]和[算法paper]适配的经验池。

`/safety_plotter`：该文件夹定义了日志文件的记载以及画图的相关设置

`plot.py` 用于将训练生成的log进行绘图。

`train_metadrive.py` ： The main script for running experiments in metadrive, which parses command-line arguments from the user .

## Installation and Setup

To install the dependencies，use the command line 

`pip instal -r requirements.txt`



## Running Experiments

我们可以通过命令行的方式运行所有的安全强化学习算法和本框架涉及到的2个环境。

你可以使用如下的方式运行实验：

`python main.py --use_td3 --env SafetyCarAvoidance-v0  --save_model  --exp_name td3 --seed 0 --loop 5 --device 1 --max_timesteps 1000000`



## Showing Results

=======
针对metadrive的不同算法的曲线的参数：
rec: soft_plus delta=0.1
qpsl:delta=0.02
lag: init=1.0 lr=1e-4
>>>>>>> 719c13747215e8fe4bc8a034fa7bcac515f0a841
