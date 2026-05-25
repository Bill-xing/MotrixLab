# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
ANYmal C 导航环境配置文件

本模块定义了 ANYmal C 四足机器人导航环境的所有配置参数，包括：
- 噪声配置：传感器观测的噪声参数
- 控制配置：动作缩放因子和控制参数
- 初始化状态：机器人初始位置和关节角度
- 命令配置：目标位置和朝向的范围
- 归一化参数：观测数据的归一化系数
- 资产配置：机器人身体部分和接触几何体
- 传感器配置：传感器名称映射
- 奖励配置：奖励函数的缩放系数
"""

import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

# 加载机器人模型文件路径（MJCF/XML格式）
model_file = os.path.dirname(__file__) + "/xmls/scene.xml"


@dataclass
class NoiseConfig:
    """
    传感器噪声配置

    用于模拟真实传感器的测量噪声，提高RL模型的鲁棒性。
    所有参数都是乘法因子，与观测值相乘得到带噪声的结果。
    """
    # 噪声启用开关（0=禁用，1=启用）
    level: float = 1.0
    # 关节角度的噪声标准差（rad）
    scale_joint_angle: float = 0.03
    # 关节速度的噪声标准差（rad/s）
    scale_joint_vel: float = 1.5
    # 陀螺仪角速度的噪声标准差（rad/s）
    scale_gyro: float = 0.2
    # 重力向量的噪声标准差（标准化单位）
    scale_gravity: float = 0.05
    # 线速度的噪声标准差（m/s）
    scale_linvel: float = 0.1


@dataclass
class ControlConfig:
    """
    机器人控制配置

    定义动作空间和力矩控制参数。
    参考go1的成功配置进行调整。
    """
    # 关节刚度[N*m/rad] - 参考go1使用80
    stiffness: float = 80.0

    # 关节阻尼[N*m*s/rad] - 参考go1使用1
    damping: float = 1.0

    # 动作缩放系数 - 将[-1, 1]的动作缩放到实际关节角度变化量（rad）
    # 实际目标角度 = 默认角度 + action * action_scale
    # 参考go1使用0.05
    action_scale: float = 0.05

    # 扭矩限制[N*m] - 由XML中的forcerange参数控制


@dataclass
class InitState:
    """
    机器人初始化状态配置

    定义环境重置时机器人的位置、姿态和关节角度。
    """
    # 机器人在世界坐标系中的初始位置 [x, y, z]（米）
    # Z轴高度应与XML中base的初始高度一致，以保证正确的接地
    pos = [0.0, 0.0, 0.5]

    # 位置随机化范围 [x_min, y_min, x_max, y_max]（米）
    # 每次环境重置时，在此范围内随机放置机器人，覆盖20m x 20m的区域
    pos_randomization_range = [-10.0, -10.0, 10.0, 10.0]

    # 默认关节角度字典
    # key: 关节名称，value: 默认角度（弧度）
    # 对应ANYmal C的12个驱动关节：
    # - 4个髋关节外展/内收(HAA): LF/RF/LH/RH_HAA
    # - 4个髋关节屈伸(HFE): LF/RF/LH/RH_HFE
    # - 4个膝关节屈伸(KFE): LF/RF/LH/RH_KFE
    # LF=Left Front(左前), RF=Right Front(右前)
    # LH=Left Hind(左后), RH=Right Hind(右后)
    default_joint_angles = {
        "LF_HAA": 0.0,   # 左前髋外展（0=中立位置）
        "RF_HAA": 0.0,   # 右前髋外展
        "LH_HAA": 0.0,   # 左后髋外展
        "RH_HAA": 0.0,   # 右后髋外展
        "LF_HFE": 0.4,   # 左前髋屈伸（正值=屈曲）
        "RF_HFE": 0.4,   # 右前髋屈伸
        "LH_HFE": -0.4,  # 左后髋屈伸（负值=伸展）
        "RH_HFE": -0.4,  # 右后髋屈伸
        "LF_KFE": -0.8,  # 左前膝屈伸（负值=膝弯曲）
        "RF_KFE": -0.8,  # 右前膝屈伸
        "LH_KFE": 0.8,   # 左后膝屈伸（正值=膝伸展）
        "RH_KFE": 0.8,   # 右后膝屈伸
    }


@dataclass
class Commands:
    """
    目标命令配置

    定义RL代理需要追踪的目标位置和朝向的范围。
    """
    # 目标位置和朝向范围
    # 格式: [dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max]
    # dx/dy: 目标相对于机器人初始位置的水平偏移（米）
    # 减小范围到 [-3m, 3m]，使目标更容易到达
    # yaw: 目标的绝对朝向（弧度），范围[-π, π]
    pose_command_range = [-3.0, -3.0, -3.14, 3.0, 3.0, 3.14]

    # 目标点与机器人之间的最大允许高度差（米）
    # 减小到0.5m，避免生成难以到达的目标点
    max_height_diff: float = 0.5

    # 目标点的最大允许坡度（弧度）
    # 约11度，避免目标点在陡坡上导致机器狗难以稳定站立
    max_target_slope: float = 0.2

    # 机器人生成位置的最大允许坡度（弧度）
    # 约15度，超过此坡度的位置会被重新采样
    max_spawn_slope: float = 0.26

    # 目标点采样最大尝试次数
    max_resample_attempts: int = 20


@dataclass
class Normalization:
    """
    观测数据归一化配置

    定义观测空间中各分量的归一化系数。
    归一化后的观测 = 原始观测 * 归一化系数
    这些系数用于将不同量纲的观测统一到合理的数值范围。
    """
    # 线速度的归一化系数（乘法因子）
    # 用于缩放 [vx, vy, vz] (m/s)
    lin_vel = 2.0
    # 角速度的归一化系数（乘法因子）
    # 用于缩放陀螺仪输出 (rad/s)
    ang_vel = 0.25
    # 关节角度的归一化系数
    # 用于缩放关节位置 (rad)
    dof_pos = 1.0
    # 关节速度的归一化系数
    # 用于缩放关节速度 (rad/s)
    dof_vel = 0.05


@dataclass
class Asset:
    """
    资产配置 - 定义机器人模型中的关键身体部分和几何体

    用于指定机器人身体部分的名称和接触检测相关的几何体。
    """
    # 机器人主体body的名称（根节点）
    body_name = "base"
    # 足部几何体的名称列表，用于脚步接地检测
    # 包括4只脚的接地传感器
    foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
    # 检测基座接触的几何体名称列表
    # 如果这些几何体与地面接触，表示机器人摔倒，应终止episode
    terminate_after_contacts_on = ["base"]
    # 地面几何体的名称
    ground_name = "ground"


@dataclass
class Sensor:
    """
    传感器配置 - 定义观测中使用的传感器名称映射

    这些传感器在XML模型文件中定义，此处指定其名称以供环境查询。
    """
    # 基座线速度传感器的名称（在XML中定义）
    base_linvel = "base_linvel"
    # 基座角速度传感器（陀螺仪）的名称（在XML中定义）
    base_gyro = "base_gyro"


@dataclass
class RewardConfig:
    """
    奖励函数配置

    定义RL训练过程中奖励函数各项的权重系数。
    参考 go1 的成功配置进行调整。
    """
    # 奖励项缩放系数字典
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # 终止条件的惩罚系数
            "termination": -0.0,
            # 速度追踪奖励（参考 go1）
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.3,
            # 步态奖励
            "feet_air_time": 1.0,
            # 惩罚项
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "torques": -0.00001,
            "action_rate": -0.001,
            # 关节位置惩罚（参考 go1 的 hip_pos）
            "haa_pos": -1.0,
            "kfe_pos": -0.3,
        }
    )

    # 速度追踪的高斯核参数（参考 go1）
    tracking_sigma: float = 0.25


# === 场景文件路径 ===
rough_model_file = os.path.dirname(__file__) + "/xmls/scene_rough.xml"


@registry.envcfg("anymal-c-flat-terrain-nav")
@registry.envcfg("anymal_c_navigation_flat")
@dataclass
class AnymalCEnvCfg(EnvCfg):
    """
    ANYmal C 导航环境的完整配置类

    @registry.envcfg("anymal_c_navigation_flat") 装饰器将此配置注册到全局环境注册表，
    环境可通过字符串ID "anymal_c_navigation_flat" 查找并使用此配置。

    主要参数说明：
    - 模拟参数: 控制仿真的时间步长和物理精度
    - 一级参数: 环保、重置、速度限制等全局约束
    - 子配置: 通过组合多个专用配置类来管理复杂的参数空间
    """
    # 机器人模型文件路径（MJCF/XML格式）
    model_file: str = model_file

    # 重置时的噪声标度因子（应用于初始位置和速度）
    reset_noise_scale: float = 0.01

    # 单个episode的最大持续时间（秒）
    max_episode_seconds: float = 10

    # 单个episode的最大步数
    # 10秒 / 0.01秒 = 1000步
    max_episode_steps: int = 1000

    # 仿真时间步长（秒），控制物理引擎的积分精度
    # 0.005秒 = 5毫秒，对应200Hz的仿真频率
    sim_dt: float = 0.005

    # 控制时间步长（秒），智能体多久执行一次新动作
    # 0.01秒 = 100Hz 控制频率，每2个仿真步执行一次动作
    ctrl_dt: float = 0.01

    # 初始朝向的随机扰动标度（弧度）
    reset_yaw_scale: float = 0.1

    # 关节速度的最大阈值（rad/s）
    # 如果关节速度超过此值，视为异常并终止episode以保护训练稳定性
    # 初期设置为100 rad/s以给予较大容忍度
    max_dof_vel: float = 100.0

    # 噪声配置实例 - 传感器测量噪声参数
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)

    # 控制配置实例 - 动作空间和控制参数
    control_config: ControlConfig = field(default_factory=ControlConfig)

    # 奖励配置实例 - 奖励函数参数（当前未使用）
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    # 初始化状态配置实例 - 机器人初始位置和关节角度
    init_state: InitState = field(default_factory=InitState)

    # 命令配置实例 - 目标位置和朝向范围
    commands: Commands = field(default_factory=Commands)

    # 归一化配置实例 - 观测数据的标准化系数
    normalization: Normalization = field(default_factory=Normalization)

    # 资产配置实例 - 机器人身体部分和几何体定义
    asset: Asset = field(default_factory=Asset)

    # 传感器配置实例 - 传感器名称映射
    sensor: Sensor = field(default_factory=Sensor)


@dataclass
class TerrainConfig:
    """
    崎岖地形配置

    定义地形网格的参数，用于机器人在不同高度区域的放置和训练。
    """
    # 地形高度级别列表（米）
    # 对应高度场中的不同区域高度
    height_list: tuple[float, ...] = (0.0, 0.5, 1.0)

    # 网格大小（5x5 网格）
    grid_size: int = 5

    # 每个网格单元的尺寸（米）
    cell_size: float = 8.0

    # 地形边界半径（米）- 超出此范围将终止 episode
    border_size: float = 19.0


@registry.envcfg("anymal_c_navigation_rough")
@dataclass
class AnymalCRoughEnvCfg(AnymalCEnvCfg):
    """ANYmal C 崎岖地形导航环境配置

    相比平面地形：
    - 使用崎岖地形场景文件
    - 渲染间距设为 0（崎岖地形环境不需要间距）
    - 延长 episode 时长（崎岖地形更具挑战性）
    - 包含地形配置参数
    """
    # 使用崎岖地形场景文件
    model_file: str = rough_model_file

    # 渲染间距设为 0（崎岖地形环境不需要间距）
    render_spacing: float = 0.0

    # 延长 episode 时长（崎岖地形更具挑战性）
    max_episode_seconds: float = 15
    max_episode_steps: int = 1500

    # 地形配置
    terrain_config: TerrainConfig = field(default_factory=TerrainConfig)
