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

"""Configuration for Anymal C Navigation Task.

该模块定义了Anymal C四足机器人在平坦地形上进行导航任务的所有配置参数。
包括机器人控制、传感器、奖励函数、命令以及物理仿真参数。
"""

import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

# 获取XML模型文件的绝对路径，该文件定义了仿真场景的物理模型
model_file = os.path.dirname(__file__) + "/xmls/scene.xml"


@dataclass
class NoiseCfg:
    """观测噪声配置。

    用于模拟传感器的噪声，增加环境的随机性，帮助训练更鲁棒的策略。
    """
    # 总体噪声级别系数，1.0表示标准噪声，0表示无噪声
    level: float = 1.0
    # 关节角度观测的噪声尺度 [rad]
    scale_joint_angle: float = 0.03
    # 关节速度观测的噪声尺度 [rad/s]
    scale_joint_vel: float = 1.5
    # 陀螺仪（角速度）观测的噪声尺度 [rad/s]
    scale_gyro: float = 0.2
    # 重力加速度观测的噪声尺度 [m/s^2]
    scale_gravity: float = 0.05
    # 线速度观测的噪声尺度 [m/s]
    scale_linvel: float = 0.1


@dataclass
class ControlCfg:
    """PD控制器配置。

    用于控制Anymal C机器人的12个关节（每条腿3个）。
    """
    # 关节刚度系数 [N*m/rad]，值越大机器人越硬，100.0对应Anymal C的较高刚度
    stiffness: float = 100.0
    # 关节阻尼系数 [N*m*s/rad]，用于减缓关节运动
    damping: float = 2.0
    # 动作缩放系数，用于缩放神经网络输出到关节位置目标值的范围
    # 值为0.25表示动作范围[-1,1]将被缩放到[-0.25,0.25]弧度
    action_scale: float = 0.25


@dataclass
class InitStateCfg:
    """初始状态配置。

    定义每个训练回合开始时机器人的位置和姿态。
    """
    # 机器人初始位置 [x, y, z]，单位为米
    # z=0.62m是机器人站立时的高度（从脚到机身中心的高度）
    pos = [0.0, 0.0, 0.62]

    # Anymal C的默认关节角度 [rad]
    # 关节命名规则: {LF|RF|LH|RH}_{HAA|HFE|KFE}
    # 其中:
    #   - LF: Left Front（左前腿）, RF: Right Front（右前腿）
    #     LH: Left Hind（左后腿）, RH: Right Hind（右后腿）
    #   - HAA: Hip Abduction/Adduction（髋关节内外展）
    #   - HFE: Hip Flexion/Extension（髋关节屈伸）
    #   - KFE: Knee Flexion/Extension（膝关节屈伸）
    default_joint_angles = {
        # 左前腿：站立姿态
        "LF_HAA": 0.0,      # 髋关节内外展中立位
        "LF_HFE": 0.4,      # 髋关节略微屈曲
        "LF_KFE": -0.8,     # 膝关节屈曲
        # 右前腿：站立姿态
        "RF_HAA": 0.0,
        "RF_HFE": 0.4,
        "RF_KFE": -0.8,
        # 左后腿：站立姿态
        "LH_HAA": 0.0,
        "LH_HFE": -0.4,     # 髋关节略微伸展
        "LH_KFE": 0.8,      # 膝关节伸展
        # 右后腿：站立姿态
        "RH_HAA": 0.0,
        "RH_HFE": -0.4,
        "RH_KFE": 0.8,
    }


@dataclass
class CommandsCfg:
    """导航命令配置。

    定义在导航任务中速度命令的范围和目标位置的范围。
    """
    # 速度命令限制 [最小值, 最大值]，分别对应 [vel_x, vel_y, ang_vel]
    # vel_x：前后方向线速度 [m/s]，范围[0, 2.0]表示只能向前或停止
    # vel_y：左右方向线速度 [m/s]，范围[-1.0, 1.0]表示可以横向移动
    # ang_vel：角速度 [rad/s]，范围[-1.0, 1.0]表示可以旋转
    vel_limit = [
        [0.0, -1.0, -1.0],  # 最小值
        [2.0, 1.0, 1.0],    # 最大值
    ]

    # 导航目标位置范围 [最小值, 最大值]
    # 定义了在仿真环境中可以生成的目标位置范围
    pos_range = [
        [-5.0, -5.0],  # 最小值：x_min [m], y_min [m]
        [5.0, 5.0],    # 最大值：x_max [m], y_max [m]
    ]

    # 命令重采样时间范围 [最小值, 最大值]，单位秒
    # 表示新的速度命令会在4-10秒的随机间隔内更新
    # 这会改变机器人需要跟踪的目标速度
    resampling_time_range = [4.0, 10.0]


@dataclass
class NormalizationCfg:
    """观测归一化系数。

    用于将观测值缩放到合理的范围，通常为[-1, 1]，
    帮助神经网络更快地训练和收敛。
    """
    # 线速度观测的归一化系数，实际观测会除以此值
    # 2.0表示最大线速度约2.0 m/s
    lin_vel: float = 2.0
    # 角速度观测的归一化系数
    # 0.25表示最大角速度约0.25 rad/s
    ang_vel: float = 0.25
    # 自由度（关节）位置的归一化系数，单位弧度
    # 1.0表示关节位置范围约[-1, 1]弧度
    dof_pos: float = 1.0
    # 自由度（关节）速度的归一化系数，单位rad/s
    # 0.05表示关节速度范围约[-0.05, 0.05]弧度/秒
    dof_vel: float = 0.05


@dataclass
class AssetCfg:
    """资产（机器人部件）配置。

    定义了机器人各个部件的名称，用于在仿真中识别和交互。
    """
    # 机器人身体（躯干）的名称，用于获取其位置和姿态信息
    body_name: str = "base"
    # 机器人脚的名称，用于检测脚与地面的接触
    foot_name: str = "foot"
    # 大腿（THIGH）和小腿（SHANK）与地面接触时会受到惩罚
    # 这鼓励机器人用脚着陆而不是用腿的其他部分着陆
    penalize_contacts_on = ["THIGH", "SHANK"]
    # 如果机器人的身体（base）与地面接触，则认为机器人摔倒，回合终止
    terminate_after_contacts_on = ["base"]
    # 地面的名称
    ground: str = "floor"


@dataclass
class SensorCfg:
    """传感器配置。

    定义了从仿真环境中获取的传感器数据的名称。
    """
    # 本地线速度传感器的名称，表示机器人相对于自身坐标系的速度
    local_linvel: str = "local_linvel"
    # 陀螺仪传感器的名称，用于获取机器人的角速度
    gyro: str = "gyro"


@dataclass
class RewardsCfg:
    """导航任务的奖励配置。

    定义了各个奖励分量及其权重，用于引导强化学习代理学习期望的行为。
    正值表示鼓励的行为，负值表示惩罚的行为。
    """
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # ==================== 导航相关奖励 ====================
            # 粗粒度位置追踪奖励：鼓励机器人到达目标位置（容差范围2.0米）
            "position_tracking": 1.0,
            # 精细位置追踪奖励：鼓励机器人精确到达目标位置（容差范围0.2米）
            "position_tracking_fine_grained": 1.5,
            # 朝向追踪惩罚：轻微惩罚不正确的朝向，帮助机器人面向运动方向
            "orientation_tracking": -0.1,

            # ==================== 运动控制相关奖励 ====================
            # 回合提前终止惩罚：机器人摔倒或发生碰撞时的重大惩罚
            "termination_penalty": -200.0,
            # 线速度追踪奖励：鼓励机器人准确跟踪目标线速度
            "tracking_lin_vel": 1.0,
            # 角速度追踪奖励：鼓励机器人准确跟踪目标角速度（权重较小）
            "tracking_ang_vel": 0.5,
            # 竖直线速度惩罚：禁止机器人在竖直方向上移动（不应该跳起或下沉）
            "lin_vel_z": -2.0,
            # 横滚俯仰角速度惩罚：鼓励机器人保持平衡，避免翻滚
            "ang_vel_xy": -0.05,
            # 姿态奖励：轻微鼓励保持竖直姿态（权重为0，实际不起作用）
            "orientation": -0.0,
            # 关节扭矩惩罚：轻微惩罚过大的关节扭矩，鼓励高效运动
            "torques": -0.00001,
            # 关节速度惩罚：轻微惩罚过高的关节速度（权重为0，实际不起作用）
            "dof_vel": -0.0,
            # 关节加速度惩罚：轻微惩罚过高的关节加速度，鼓励平滑运动
            "dof_acc": -2.5e-7,
            # 脚脱离地面时间奖励：鼓励良好的步态，即脚要有合理的脱离地面时间
            "feet_air_time": 1.0,
            # 动作变化率惩罚：轻微惩罚动作突变，鼓励平滑的控制输出
            "action_rate": -0.001,
            # 静止惩罚：不鼓励机器人静止不动（权重为0，实际不起作用）
            "stand_still": -0.0,
            # 髋关节位置惩罚：鼓励机器人腿部不过度伸展或缩回
            "hip_pos": -0.5,
        }
    )

    # 速度追踪的标准差，用于高斯函数计算奖励
    tracking_sigma: float = 0.25
    # 粗粒度位置追踪的标准差 [米]，控制位置误差的容差范围
    position_tracking_std: float = 2.0
    # 精细位置追踪的标准差 [米]，更严格的位置误差容差范围
    position_tracking_fine_grained_std: float = 0.2


@registry.envcfg("anymal-c-flat-terrain-nav")
@dataclass
class AnymalCNavEnvCfg(EnvCfg):
    """Anymal C平坦地形导航环境的配置。

    该配置类继承自EnvCfg基类，包含了完整的环境设置，包括：
    - 物理仿真参数
    - 机器人控制配置
    - 传感器配置
    - 奖励函数设置
    - 观测和命令的定义

    环境注册为 "anymal-c-flat-terrain-nav"，可通过该名称加载使用。
    """
    # 每个训练回合的最大持续时间 [秒]
    max_episode_seconds: float = 10.0
    # XML模型文件的路径，定义了机器人和场景的物理属性
    model_file: str = model_file
    # 观测噪声配置实例
    noise_cfg: NoiseCfg = field(default_factory=NoiseCfg)
    # PD控制器配置实例
    control_cfg: ControlCfg = field(default_factory=ControlCfg)
    # 奖励函数配置实例
    rewards_cfg: RewardsCfg = field(default_factory=RewardsCfg)
    # 初始状态配置实例
    init_state_cfg: InitStateCfg = field(default_factory=InitStateCfg)
    # 导航命令配置实例
    commands_cfg: CommandsCfg = field(default_factory=CommandsCfg)
    # 观测归一化配置实例
    normalization_cfg: NormalizationCfg = field(default_factory=NormalizationCfg)
    # 机器人资产配置实例
    asset_cfg: AssetCfg = field(default_factory=AssetCfg)
    # 传感器配置实例
    sensor_cfg: SensorCfg = field(default_factory=SensorCfg)
    # 仿真时间步长 [秒]，0.005表示200 Hz的仿真频率
    sim_dt: float = 0.005
    # 控制时间步长 [秒]，0.02表示50 Hz的控制频率
    # 这意味着每4个仿真步长会执行一次控制命令
    ctrl_dt: float = 0.02
