import os
import numpy as np
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/xmls/scene.xml"


@dataclass
class ControlConfig:
    # 位置控制
    # mjcf 中 actuator 定义的是 <position ..../>
    actuators = ["actuator1", "actuator2", "actuator3", "actuator4", "actuator5", "actuator6", "actuator7", "actuator8"]
    joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1"]
    min_pos = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8972, 0]
    max_pos = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8972, 0.04]

    joint_pos_reset_noise = 0.125


@registry.envcfg("franka_open_cabinet")
@dataclass
class FrankaOpenCabinetEnvCfg(EnvCfg):
    model_file: str = model_file
    max_episode_seconds: float = 5.0
    sim_dt: float = 0.01
    move_speed: float = 1.0
    ctrl_dt: float = 0.01


    control_config: ControlConfig = field(default_factory=ControlConfig)
