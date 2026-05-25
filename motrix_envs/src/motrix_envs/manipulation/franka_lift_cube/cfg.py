import os
import numpy as np
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/xmls/mjx_scene.xml"

@dataclass
class InitState:
    # robot joint names and default positions [rad]
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'finger_joint1', 'finger_joint2']
    default_joint_pos = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04], np.float32)

    joint_pos_reset_noise = 0.125
    cube_pos_x_reset_noise = [-0.1, 0.1]
    cube_pos_y_reset_noise = [-0.25, 0.25]


@dataclass
class ControlConfig:
    # 位置控制
    # xml文件 actuator 定义的是 <position ..../>
    actuators = ["actuator1", "actuator2", "actuator3", "actuator4", "actuator5", "actuator6", "actuator7", "actuator8"]
    min_pos = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -np.pi/2, 0]
    max_pos = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, np.pi/2, 0.04]


@dataclass
class Commands:

    target_pos_x = [0.4, 0.6]
    target_pos_y = [-0.25, 0.25]
    target_pos_z = [0.25, 0.5]


@registry.envcfg("franka_lift_cube")
@dataclass
class FrankaLiftCubeEnvCfg(EnvCfg):
    model_file: str = model_file
    max_episode_seconds: float = 2.5
    sim_dt: float = 0.01
    move_speed: float = 1.0
    ctrl_dt: float = 0.01


    init_state: InitState = field(default_factory=InitState)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    command_config: Commands = field(default_factory=Commands)
