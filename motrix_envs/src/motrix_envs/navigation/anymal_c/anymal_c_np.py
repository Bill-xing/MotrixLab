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
ANYmal C 四足机器人导航环境实现

该模块实现了基于 MotrixSim 物理引擎的 ANYmal C 四足机器人导航任务环境。
机器人需要学习在平坦地形上导航到指定目标位置和朝向。

主要功能：
- 位置和朝向跟踪任务
- 速度命令生成与跟踪
- 接触检测和终止条件
- 奖励函数设计
- 可视化标记（目标位置、运动方向箭头）
"""

import logging
import os

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import AnymalCEnvCfg, AnymalCRoughEnvCfg, ControlConfig

logger = logging.getLogger(__name__)


@registry.env("anymal-c-flat-terrain-nav", "np")
@registry.env("anymal_c_navigation_flat", "np")
class AnymalCEnv(NpEnv):
    """
    ANYmal C 四足机器人导航环境

    该环境实现了一个导航任务，机器人需要：
    1. 移动到指定的目标位置（x, y）
    2. 调整到指定的目标朝向（yaw）
    3. 在目标位置停稳（速度和角速度接近零）

    观测空间（54维）：
    - 线速度（3）：机身在世界坐标系下的线速度
    - 角速度（3）：机身的角速度（陀螺仪）
    - 投影重力（3）：重力在机身坐标系的投影
    - 关节位置（12）：12个关节的角度（相对默认角度）
    - 关节速度（12）：12个关节的角速度
    - 上一步动作（12）：上一时刻的动作
    - 速度命令（3）：期望的线速度和角速度命令
    - 位置误差（2）：到目标的位置误差向量
    - 朝向误差（1）：到目标的朝向误差
    - 距离（1）：到目标的距离
    - 到达标志（1）：是否到达目标
    - 停止就绪标志（1）：是否满足停止条件

    动作空间（12维）：
    - 12个关节的目标位置偏移（相对默认角度）
    - 范围：[-1, 1]，经过 action_scale 缩放
    """
    _cfg: AnymalCEnvCfg

    def __init__(self, cfg:AnymalCEnvCfg, num_envs: int = 1):
        """
        初始化 ANYmal C 导航环境

        参数：
            cfg: 环境配置对象，包含机器人参数、奖励权重、初始状态等
            num_envs: 并行环境数量（用于向量化训练）
        """
        # 调用父类初始化，创建物理模型和场景
        super().__init__(cfg, num_envs = num_envs)

        # 获取机器人主体（用于查询位置、速度等状态）
        # 获取机器人主体（用于查询位置、速度等状态）
        self._body = self._model.get_body(cfg.asset.body_name)
        # 初始化接触检测几何体（足部和基座）
        self._init_contact_geometry()

        # 获取目标标记的body（红色圆柱，用于可视化目标位置）
        self._target_marker_body = self._model.get_body("target_marker")

        # 获取箭头body（用于可视化运动方向，不影响物理仿真）
        try:
            # 绿色箭头：机器人当前运动方向
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            # 蓝色箭头：期望运动方向
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception as e:
            # 如果模型中没有箭头，设为None（不影响训练）
            self._robot_arrow_body = None
            self._desired_arrow_body = None

        # 动作空间：12个关节的目标位置偏移，范围 [-1, 1]
        # 动作空间：12个关节的目标位置偏移，范围 [-1, 1]
        self._action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape = (12,), dtype = np.float32)
        # 观测空间：54维向量，包含本体感知、任务信息等
        # 组成：linvel(3) + gyro(3) + gravity(3) + joint_pos(12) + joint_vel(12) +
        #       last_actions(12) + commands(3) + position_error(2) + heading_error(1) +
        #       distance(1) + reached_flag(1) + stop_ready_flag(1) = 54
        self._observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (54,), dtype = np.float32)

        # 自由度相关参数
        self._num_dof_pos = self._model.num_dof_pos  # 位置自由度数量
        self._num_dof_vel = self._model.num_dof_vel  # 速度自由度数量
        self._num_action = self._model.num_actuators  # 执行器（关节）数量

        # 初始化DOF状态（位置和速度）
        self._init_dof_pos = self._model.compute_init_dof_pos()  # 从模型计算初始位置
        self._init_dof_vel = np.zeros(
            (self._model.num_dof_vel,),
            dtype=np.float32,
        )  # 初始速度为零

        # 查找并设置target_marker的DOF索引，更新初始位置
        self._find_target_marker_dof_indices()

        # 查找箭头的DOF索引（如果箭头body存在）
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()

        # 初始化环境缓冲区（默认角度、归一化系数等）
        self._init_buffer()

    def _init_buffer(self):
        """
        初始化环境缓冲区

        设置：
        - 默认关节角度（站立姿态）
        - 归一化系数（用于观测归一化）
        - 初始DOF位置
        - 关节索引（HAA/KFE）
        - PD控制参数
        """
        cfg = self._cfg
        # 初始化默认关节角度数组（12个关节）
        self.default_angles = np.zeros(self._num_action, dtype = np.float32)

        # PD控制器参数（参考go1的配置）
        self.kps = np.ones(self._num_action, dtype=np.float32) * cfg.control_config.stiffness
        self.kds = np.ones(self._num_action, dtype=np.float32) * cfg.control_config.damping

        # 归一化系数：用于将物理量归一化到合理范围
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )  # [vx缩放, vy缩放, vyaw缩放]

        # 关节索引（用于奖励计算）
        self.haa_indices = []  # 髋关节外展/内收
        self.kfe_indices = []  # 膝关节屈伸

        # 从配置中读取默认关节角度（站立姿态）
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                # 如果关节名称匹配，设置对应角度
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle
            # 识别关节类型
            actuator_name = self._model.actuator_names[i]
            if "HAA" in actuator_name:
                self.haa_indices.append(i)
            if "KFE" in actuator_name:
                self.kfe_indices.append(i)

        # 更新初始DOF位置的关节部分（最后12个DOF对应关节）
        self._init_dof_pos[-self._num_action:] = self.default_angles

    def _find_target_marker_dof_indices(self):
        """
        查找target_marker在dof_pos中的索引位置

        DOF结构布局（Motrix引擎）：
        - DOF 0-3: target_marker (slide x, slide y, slide z, hinge yaw) - 目标标记的位置和朝向
        - DOF 4-6: base position (x, y, z) - 机器人基座的世界坐标位置
        - DOF 7-10: base quaternion (qx, qy, qz, qw) - 机器人基座的姿态（Motrix格式）
        - DOF 11+: joint angles (12个关节角度)

        该方法设置：
        - target_marker的DOF索引范围
        - target_marker的初始位置（原点，朝向为0）
        - base四元数的DOF索引范围
        """
        # target_marker占用前4个DOF (x, y, z, yaw)
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 4

        # 设置target_marker的初始位置为原点，朝向为0
        self._init_dof_pos[0:4] = [0.0, 0.0, 0.0, 0.0]  # [x, y, z, yaw]

        # 记录base四元数的DOF索引（用于后续归一化）
        self._base_quat_start = 7
        self._base_quat_end = 11

    def _find_arrow_dof_indices(self):
        """
        查找箭头在dof_pos中的索引位置

        完整DOF结构：
        - DOF 0-3: target_marker (4个: slide x, slide y, slide z, hinge yaw)
        - DOF 4-6: base position (3个: x, y, z)
        - DOF 7-10: base quaternion (4个: qx, qy, qz, qw)
        - DOF 11-22: joint angles (12个关节)
        - DOF 23-29: robot_heading_arrow freejoint (7个: 3 pos + 4 quat) - 绿色箭头
        - DOF 30-36: desired_heading_arrow freejoint (7个: 3 pos + 4 quat) - 蓝色箭头

        箭头用于可视化，不参与物理仿真，每个箭头有7个DOF（位置3 + 四元数4）
        """
        # robot_heading_arrow（绿色）的DOF索引范围
        self._robot_arrow_dof_start = 23
        self._robot_arrow_dof_end = 30

        # desired_heading_arrow（蓝色）的DOF索引范围
        self._desired_arrow_dof_start = 30
        self._desired_arrow_dof_end = 37

        # 设置robot_heading_arrow的初始位置和姿态: [x, y, z, qx, qy, qz, qw]
        # z=0.76是箭头高度（base高度0.56 + 0.2偏移）
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [0.0, 0.0, 0.76, 0.0, 0.0, 0.0, 1.0]

        # 设置desired_heading_arrow的初始位置和姿态
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [0.0, 0.0, 0.76, 0.0, 0.0, 0.0, 1.0]

    def _init_contact_geometry(self):
        """
        初始化接触检测所需的几何体索引

        设置两类接触检测：
        1. 终止接触：基座与地面接触（触发终止）
        2. 足部接触：用于步态分析和奖励计算
        """
        cfg = self._cfg
        # 获取地面几何体索引
        self.ground_index = self._model.get_geom_index(cfg.asset.ground_name)

        # 初始化两类接触检测矩阵
        self._init_termination_contact()  # 基座接触检测
        self._init_foot_contact()         # 足部接触检测

    def _init_termination_contact(self):
        """
        初始化终止接触检测

        检测机器人基座是否与地面接触。如果基座触地，说明机器人摔倒，
        应该终止该episode。

        创建检测对：[基座几何体索引, 地面几何体索引]
        """
        cfg = self._cfg
        # 查找所有需要检测的基座几何体
        base_indices = []
        for base_name in cfg.asset.terminate_after_contacts_on:
            try:
                base_idx = self._model.get_geom_index(base_name)
                if base_idx is not None:
                    base_indices.append(base_idx)
                else:
                    logger.warning("Geom %r not found in model", base_name)
            except Exception as e:
                logger.warning("Error finding base geom %r: %s", base_name, e)

        # 创建基座-地面接触检测矩阵 (N x 2)
        # 每行是一对 [基座索引, 地面索引]
        if base_indices:
            self.termination_contact = np.array(
                [[idx, self.ground_index] for idx in base_indices],
                dtype=np.uint32
            )
            self.num_termination_check = self.termination_contact.shape[0]
        else:
            # 如果没有配置，使用空数组
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            logger.warning("No base contacts configured for termination")

    def _init_foot_contact(self):
        """
        初始化足部接触检测

        检测四个足部是否与地面接触。用于：
        - 步态分析
        - 接触相关的奖励计算
        - 监控机器人行走状态

        创建检测对：[足部几何体索引, 地面几何体索引]
        """
        cfg = self._cfg
        foot_indices = []
        # 查找所有足部几何体
        for foot_name in cfg.asset.foot_names:
            try:
                foot_idx = self._model.get_geom_index(foot_name)
                if foot_idx is not None:
                    foot_indices.append(foot_idx)
                else:
                    logger.warning("Foot geom %r not found in model", foot_name)
            except Exception as e:
                logger.warning("Error finding foot geom %r: %s", foot_name, e)

        # 创建足部-地面接触检测矩阵 (N x 2)
        if foot_indices:
            self.foot_contact_check = np.array(
                [[idx, self.ground_index] for idx in foot_indices],
                dtype=np.uint32
            )
            self.num_foot_check = self.foot_contact_check.shape[0]
        else:
            self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
            self.num_foot_check = 0
            logger.warning("No foot contacts configured")

    def get_dof_pos(self, data: mtx.SceneData):
        """获取关节DOF位置（12个关节角度）"""
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData):
        """获取关节DOF速度（12个关节角速度）"""
        return self._body.get_joint_dof_vel(data)

    def _extract_root_state(self, data):
        """
        从self._body中提取根节点（基座）状态

        返回：
            root_pos: [num_envs, 3] - 基座在世界坐标系的位置 [x, y, z]
            root_quat: [num_envs, 4] - 基座姿态四元数 [qx, qy, qz, qw] (Motrix格式)
            root_linvel: [num_envs, 3] - 基座线速度（从传感器获取）
        """
        pose = self._body.get_pose(data)
        # 位置 [x, y, z]
        root_pos = pose[:, :3]
        # 四元数 [qx, qy, qz, qw] - Motrix引擎格式
        root_quat = pose[:, 3:7]
        # 使用传感器获取速度（更准确）
        root_linvel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_linvel

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        """
        应用动作到环境

        将策略输出的动作转换为执行器控制信号。
        使用位置控制模式：action表示相对默认角度的偏移。

        参数：
            actions: [num_envs, 12] - 策略输出的归一化动作 [-1, 1]
            state: 环境状态对象

        返回：
            更新后的state，其中actuator_ctrls已设置为目标关节位置
        """
        # 保存当前action用于增量控制和观测
        if "current_action" not in state.info:
            state.info["current_actions"] = np.zeros_like(actions)
        # 保存上一步的动作（用于动作平滑度惩罚）
        # 保存当前action用于增量控制
        if "current_action" not in state.info:
            state.info["current_actions"] = np.zeros_like(actions)
        state.info['last_actions'] = state.info['current_actions']
        state.info['current_actions'] = actions

        # 计算目标关节位置并设置到执行器控制
        state.data.actuator_ctrls = self._compute_torques(actions, state.data)
        return state

    def _compute_torques(self, actions, data):
        """
        计算执行器控制信号（位置控制模式）

        anymal_c 使用 position actuator，需要输出目标关节角度。
        MuJoCo 的 position actuator 会根据 XML 中配置的 kp 和 kv 自动计算力矩。

        参数：
            actions: [num_envs, 12] - 归一化动作 [-1, 1]
            data: 场景数据

        返回：
            target_pos: [num_envs, 12] - 目标关节角度
        """
        # action 表示相对于默认角度的偏移
        actions_scaled = actions * self._cfg.control_config.action_scale

        # 目标关节角 = 默认角度 + 动作偏移
        target_pos = self.default_angles + actions_scaled

        return target_pos

    def update_state(self, state:NpEnvState):
        """
        更新环境状态

        该方法在每个仿真步之后调用，完成以下任务：
        1. 从物理引擎提取最新状态（位置、速度、关节状态等）
        2. 计算期望速度命令（基于位置和朝向误差）
        3. 组装观测向量
        4. 计算奖励
        5. 判断终止条件
        6. 更新可视化标记

        参数：
            state: 当前环境状态

        返回：
            更新后的state，包含新的obs、reward、terminated
        """
        data = state.data

        # ==================== 1. 提取机器人状态 ====================
        # 获取根节点（基座）状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)

        # 关节状态（腿部关节，12个）
        joint_pos = self.get_dof_pos(data)        # [num_envs, 12] 关节角度
        joint_vel = self.get_dof_vel(data)        # [num_envs, 12] 关节角速度
        joint_pos_rel = joint_pos - self.default_angles  # 相对默认角度的偏移

        # 获取传感器数据
        base_lin_vel = root_vel[:, :3]  # 基座线速度
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)  # 陀螺仪（角速度）
        projected_gravity = self._compute_projected_gravity(root_quat)  # 重力在机身坐标系的投影

        # ==================== 2. 计算速度命令（位置追踪控制器）====================
        # 获取目标位姿命令
        pose_commands = state.info["pose_commands"]  # [num_envs, 3] (target_x, target_y, target_yaw)
        robot_position = root_pos[:, :2]  # 当前位置 [x, y]
        robot_heading = self._get_heading_from_quat(root_quat)  # 当前朝向（yaw角）
        target_position = pose_commands[:, :2]  # 目标位置
        target_heading = pose_commands[:, 2]  # 目标朝向

        # 计算位置误差和距离
        position_error = target_position - robot_position  # [num_envs, 2]
        distance_to_target = np.linalg.norm(position_error, axis=1)  # [num_envs]

        # 判断是否到达位置（阈值0.3米）
        position_threshold = 0.3
        reached_position = distance_to_target < position_threshold  # [num_envs]

        # 期望线速度（简单P控制器，比例系数1.0）
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)  # 限制最大速度
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)  # 到达后速度为0

        # 计算朝向误差（归一化到[-π, π]）
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)

        # 判断是否到达朝向（阈值15度）
        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold  # [num_envs]
        reached_all = np.logical_and(reached_position, reached_heading)  # 同时到达位置和朝向

        # 期望角速度（P控制器 + 死区）
        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)  # 死区：±8度内不输出角速度
        desired_yaw_rate = np.where(np.abs(heading_diff) < deadband_yaw, 0.0, desired_yaw_rate)

        # 到达后归零所有命令
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)



        # 组合为速度命令
        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )

        # 归一化观测
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]

        # 计算任务相关观测
        position_error_normalized = position_error / 5.0  # 归一化到合理范围
        heading_error_normalized = heading_diff / np.pi  # 归一化到[-1, 1]
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)  # 归一化距离
        reached_flag = reached_all.astype(np.float32)  # 是否到达目标

        # 计算是否达到zero_ang标准：到达且角速度接近零
        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)

        obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3
                position_error_normalized,  # 2 - 到目标的位置误差向量
                heading_error_normalized[:, np.newaxis],  # 1 - 朝向误差
                distance_normalized[:, np.newaxis],  # 1 - 到目标的距离
                reached_flag[:, np.newaxis],  # 1 - 是否已到达
                stop_ready_flag[:, np.newaxis],  # 1 - 是否达到停止标准
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 54)

        # 更新目标位置标记
        self._update_target_marker(data, pose_commands)
        # 更新箭头可视化（不影响物理）
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)

        # ==================== 更新足部接触和 feet_air_time ====================
        cquerys = self._model.get_contact_query(data)
        foot_contact = cquerys.is_colliding(self.foot_contact_check)
        foot_contact = foot_contact.reshape((self._num_envs, self.num_foot_check))
        state.info["contacts"] = foot_contact

        # 更新 feet_air_time（参考 go1 的实现）
        feet_air_time = state.info.get("feet_air_time", np.zeros((self._num_envs, self.num_foot_check), dtype=np.float32))
        feet_air_time += self._cfg.ctrl_dt
        feet_air_time *= ~foot_contact  # 接触地面时重置为0
        state.info["feet_air_time"] = feet_air_time

        # 计算奖励
        reward = self._compute_reward(data, state.info, velocity_commands)

        # 计算终止条件
        terminated_state = self._compute_terminated(state)
        terminated = terminated_state.terminated

        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        # 调试打印（每200步一次）
        state.info["steps"] = state.info.get("steps", np.zeros(self._num_envs, dtype=np.int32)) + 1
        if state.info["steps"][0] % 200 == 0:
            robot_position = root_pos[:, :2]
            robot_heading = self._get_heading_from_quat(root_quat)
            target_position = pose_commands[:, :2]
            target_heading = pose_commands[:, 2]
            position_error = np.linalg.norm(target_position - robot_position, axis=1)
            heading_diff = target_heading - robot_heading
            heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
            heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
            mean_pos_err = np.mean(position_error)
            mean_heading_err = np.rad2deg(np.mean(np.abs(heading_diff)))
            mean_vel = np.mean(np.linalg.norm(base_lin_vel[:, :2], axis=1))


        return state

    def _get_heading_from_quat(self, quat:np.ndarray) -> np.ndarray:
        """
        从四元数计算yaw角（朝向角）

        将四元数姿态转换为绕Z轴的旋转角（yaw角），用于导航任务中的朝向控制。

        参数：
            quat: [num_envs, 4] - Motrix格式四元数 [qx, qy, qz, qw]

        返回：
            heading: [num_envs] - yaw角（弧度），范围 [-π, π]
        """
        # Motrix引擎格式: [qx, qy, qz, qw]
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        # 计算yaw角（绕Z轴旋转）- 使用四元数转欧拉角公式
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return heading

    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        """
        更新箭头位置（使用DOF控制freejoint，不影响物理仿真）

        在仿真中显示两个箭头用于可视化：
        - 绿色箭头：机器人当前运动方向（实际线速度方向）
        - 蓝色箭头：期望运动方向（期望线速度方向）

        箭头通过freejoint连接，仅用于可视化，不参与碰撞检测。

        参数：
            data: 场景数据
            robot_pos: [num_envs, 3] - 机器人位置（世界坐标）
            desired_vel_xy: [num_envs, 2] - 期望线速度XY分量（地面坐标）
            base_lin_vel_xy: [num_envs, 2] - 实际线速度XY分量（地面坐标）
        """
        # 如果模型中没有箭头body，直接返回
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return

        num_envs = data.shape[0]
        arrow_height = 0.76  # 箭头高度（base高度0.56 + 0.2偏移）

        # 获取所有环境的dof_pos
        all_dof_pos = data.dof_pos.copy()

        for env_idx in range(num_envs):
            # ========== 绿色箭头：当前运动方向 ==========
            cur_v = base_lin_vel_xy[env_idx]
            if np.linalg.norm(cur_v) > 1e-3:  # 如果有明显运动
                cur_yaw = np.arctan2(cur_v[1], cur_v[0])  # 计算运动方向角
            else:
                cur_yaw = 0.0  # 静止时默认朝向
            # 箭头位置：跟随机器人
            robot_arrow_pos = np.array([
                robot_pos[env_idx, 0],
                robot_pos[env_idx, 1],
                arrow_height
            ], dtype=np.float32)
            # 箭头姿态：朝向运动方向
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            # 归一化四元数
            quat_norm = np.linalg.norm(robot_arrow_quat)
            if quat_norm > 1e-6:
                robot_arrow_quat = robot_arrow_quat / quat_norm
            # 设置DOF: [x, y, z, qx, qy, qz, qw]
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                robot_arrow_pos, robot_arrow_quat
            ])

            # ========== 蓝色箭头：期望运动方向 ==========
            des_v = desired_vel_xy[env_idx]
            if np.linalg.norm(des_v) > 1e-3:  # 如果有期望运动
                des_yaw = np.arctan2(des_v[1], des_v[0])  # 计算期望方向角
            else:
                des_yaw = 0.0  # 无期望运动时默认朝向
            # 箭头位置：跟随机器人
            desired_arrow_pos = np.array([
                robot_pos[env_idx, 0],
                robot_pos[env_idx, 1],
                arrow_height
            ], dtype=np.float32)
            # 箭头姿态：朝向期望方向
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            # 归一化四元数
            quat_norm = np.linalg.norm(desired_arrow_quat)
            if quat_norm > 1e-6:
                desired_arrow_quat = desired_arrow_quat / quat_norm
            # 设置DOF: [x, y, z, qx, qy, qz, qw]
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                desired_arrow_pos, desired_arrow_quat
            ])

        # 一次性设置所有环境的dof_pos
        data.set_dof_pos(all_dof_pos, self._model)
        # 调用正向运动学更新箭头的世界位置
        self._model.forward_kinematic(data)

    def _quat_multiply(self, q1, q2):
        """
        Motrix格式四元数乘法 [qx, qy, qz, qw]

        计算两个四元数的乘积，用于组合旋转。

        参数：
            q1: [4] - 第一个四元数 [qx, qy, qz, qw]
            q2: [4] - 第二个四元数 [qx, qy, qz, qw]

        返回：
            result: [4] - 乘积四元数 [qx, qy, qz, qw]
        """
        qx1, qy1, qz1, qw1 = q1[0], q1[1], q1[2], q1[3]
        qx2, qy2, qz2, qw2 = q2[0], q2[1], q2[2], q2[3]

        # 四元数乘法公式
        qw = qw1*qw2 - qx1*qx2 - qy1*qy2 - qz1*qz2
        qx = qw1*qx2 + qx1*qw2 + qy1*qz2 - qz1*qy2
        qy = qw1*qy2 - qx1*qz2 + qy1*qw2 + qz1*qx2
        qz = qw1*qz2 + qx1*qy2 - qy1*qx2 + qz1*qw2

        return np.array([qx, qy, qz, qw], dtype=np.float32)

    def _euler_to_quat(self, roll, pitch, yaw):
        """
        欧拉角转四元数 [qx, qy, qz, qw] - Motrix格式

        将ZYX顺序的欧拉角转换为四元数表示。

        参数：
            roll: 绕X轴旋转角（弧度）
            pitch: 绕Y轴旋转角（弧度）
            yaw: 绕Z轴旋转角（弧度）

        返回：
            quat: [4] - 四元数 [qx, qy, qz, qw]
        """
        # 计算半角的三角函数值
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        # 欧拉角到四元数转换公式（ZYX顺序）
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return np.array([qx, qy, qz, qw], dtype=np.float32)

    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray) -> np.ndarray:
        """
        计算奖励函数

        该奖励函数设计用于导航任务，包含以下主要组成部分：

        1. 速度跟踪奖励：激励机器人跟踪期望的线速度和角速度
        2. 接近奖励：激励机器人靠近目标位置
        3. 到达奖励：一次性奖励，首次到达目标时给予
        4. 停止奖励：激励机器人在目标位置停稳
        5. 惩罚项：
           - 终止条件惩罚（速度超限、基座接地、侧翻）
           - Z轴线速度惩罚（不希望垂直方向大幅运动）
           - XY轴角速度惩罚（不希望Roll/Pitch晃动）
           - 力矩惩罚（能耗最小化）
           - 关节速度惩罚（运动平滑性）
           - 动作变化惩罚（控制平滑性）

        参数：
            data: 场景数据
            info: 环境信息字典
            velocity_commands: [num_envs, 3] - 期望速度命令 (vx, vy, vyaw)

        返回：
            reward: [num_envs] - 每个环境的奖励值
        """
        # ==================== 1. 终止条件惩罚 ====================
        termination_penalty = np.zeros(self._num_envs, dtype=np.float32)

        # 检查DOF速度是否超限（关节速度过大或数值发散）
        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)  # 每个环境的最大关节速度
        vel_overflow = vel_max > self._cfg.max_dof_vel  # 是否超过配置的最大速度
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)  # 数值异常检测
        termination_penalty = np.where(vel_overflow | vel_extreme, -20.0, termination_penalty)

        # 机器人基座接触地面惩罚（机器人摔倒）
        cquerys = self._model.get_contact_query(data)
        termination_check = cquerys.is_colliding(self.termination_contact)
        termination_check = termination_check.reshape((self._num_envs, self.num_termination_check))
        base_contact = termination_check.any(axis=1)  # 是否有任何基座部分触地
        termination_penalty = np.where(base_contact, -20.0, termination_penalty)

        # 侧翻惩罚（倾斜角度>75°）
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        proj_g = self._compute_projected_gravity(root_quat)  # 投影重力
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)  # 水平分量
        gz = proj_g[:, 2]  # 垂直分量
        tilt_angle = np.arctan2(gxy, np.abs(gz))  # 倾斜角度
        side_flip_mask = tilt_angle > np.deg2rad(75)  # 是否侧翻
        termination_penalty = np.where(side_flip_mask, -20.0, termination_penalty)

        # ==================== 2. 速度跟踪奖励 ====================
        # 线速度跟踪奖励（高斯核函数）
        base_lin_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        lin_vel_error = np.sum(np.square(velocity_commands[:, :2] - base_lin_vel[:, :2]), axis=1)  # L2误差
        tracking_lin_vel = np.exp(-lin_vel_error / 0.25)  # tracking_sigma = 0.25，误差越小奖励越高

        # 角速度跟踪奖励（高斯核函数）
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        ang_vel_error = np.square(velocity_commands[:, 2] - gyro[:, 2])  # yaw角速度误差
        tracking_ang_vel = np.exp(-ang_vel_error / 0.25)

        # ==================== 3. 位置跟踪相关奖励 ====================
        # 获取机器人当前位置和朝向
        robot_position = pose[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = info["pose_commands"][:, :2]
        target_heading = info["pose_commands"][:, 2]

        # 计算位置和朝向误差
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        heading_diff = target_heading - robot_heading
        # 归一化朝向误差到[-π, π]
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)

        # 判断是否到达目标（位置<0.3m，朝向<15°）
        position_threshold = 0.3
        reached_position = distance_to_target < position_threshold
        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold
        reached_all = np.logical_and(reached_position, reached_heading)

        # 首次到达的一次性奖励（+10分）
        info["ever_reached"] = info.get("ever_reached", np.zeros(self._num_envs, dtype=bool))
        first_time_reach = np.logical_and(reached_all, ~info["ever_reached"])
        info["ever_reached"] = np.logical_or(info["ever_reached"], reached_all)
        arrival_bonus = np.where(first_time_reach, 10.0, 0.0)

        # 接近奖励：激励机器人不断靠近目标
        # 使用历史最近距离来计算进步
        if "min_distance" not in info:
            info["min_distance"] = distance_to_target.copy()
        distance_improvement = info["min_distance"] - distance_to_target  # 距离减少量
        info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)
        approach_reward = np.clip(distance_improvement * 4.0, -1.0, 1.0)  # 每接近1米奖励4分

        # 姿态稳定性奖励（惩罚偏离正常站立姿态）
        # 正常站立时 projected_gravity ≈ [0, 0, -1]
        projected_gravity = self._compute_projected_gravity(root_quat)
        orientation_penalty = np.square(projected_gravity[:, 0]) + np.square(projected_gravity[:, 1]) + np.square(projected_gravity[:, 2] + 1.0)

        # ==================== 4. 停止奖励（到达后）====================
        # 计算停止奖励：激励机器人在目标位置停稳
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)  # 平面速度
        zero_ang_mask = np.abs(gyro[:, 2]) < 0.05  # 角速度是否接近零（<0.05 rad/s ≈ 2.86°/s）
        zero_ang_bonus = np.where(np.logical_and(reached_all, zero_ang_mask), 6.0, 0.0)  # 停稳额外奖励
        # 停止基础奖励（速度和角速度都接近零时奖励更高）
        stop_base = 2 * (0.8 * np.exp(- (speed_xy / 0.2)**2) + 1.2 * np.exp(- (np.abs(gyro[:, 2]) / 0.1)**4))
        stop_bonus = np.where(reached_all, stop_base + zero_ang_bonus, 0.0)

        # ==================== 5. 各项惩罚 ====================
        # Z轴线速度惩罚（不希望垂直方向大幅运动）
        lin_vel_z_penalty = np.square(base_lin_vel[:, 2])

        # XY轴角速度惩罚（不希望Roll/Pitch晃动）
        ang_vel_xy_penalty = np.sum(np.square(gyro[:, :2]), axis=1)

        # 力矩惩罚（能耗最小化）
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=1)

        # 关节速度惩罚（运动平滑性）
        joint_vel = self.get_dof_vel(data)
        dof_vel_penalty = np.sum(np.square(joint_vel), axis=1)

        # 动作变化惩罚
        action_diff = info["current_actions"] - info["last_actions"]
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)

        # ==================== 6. go1 风格奖励（新增）====================
        # feet_air_time 奖励：鼓励自然步态，参考 go1 的实现
        feet_air_time = info.get("feet_air_time", np.zeros((self._num_envs, self.num_foot_check), dtype=np.float32))
        contacts = info.get("contacts", np.zeros((self._num_envs, self.num_foot_check), dtype=bool))
        first_contact = (feet_air_time > 0.0) * contacts
        # 奖励长步态（空中时间超过 0.5 秒才有奖励）
        feet_air_time_reward = np.sum((feet_air_time - 0.5) * first_contact, axis=1)
        # 只在有移动命令时才给予奖励
        has_movement = np.linalg.norm(velocity_commands[:, :2], axis=1) > 0.1
        feet_air_time_reward *= has_movement

        # HAA（髋关节外展）位置惩罚：鼓励髋关节保持在默认位置附近
        joint_pos = self.get_dof_pos(data)
        if len(self.haa_indices) > 0:
            haa_pos_penalty = np.sum(
                np.square(joint_pos[:, self.haa_indices] - self.default_angles[self.haa_indices]),
                axis=1
            )
        else:
            haa_pos_penalty = np.zeros(self._num_envs, dtype=np.float32)

        # KFE（膝关节）位置惩罚
        if len(self.kfe_indices) > 0:
            kfe_pos_penalty = np.sum(
                np.square(joint_pos[:, self.kfe_indices] - self.default_angles[self.kfe_indices]),
                axis=1
            )
        else:
            kfe_pos_penalty = np.zeros(self._num_envs, dtype=np.float32)

        # 综合奖励
        # 到达后：停止所有正向奖励，只保留停止奖励和惩罚项
        reward = np.where(
            reached_all,
            # 到达后：只有停止奖励和惩罚
            (
                stop_bonus
                + arrival_bonus
                - 2.0 * lin_vel_z_penalty
                - 0.05 * ang_vel_xy_penalty
                - 0.0 * orientation_penalty
                - 0.00001 * torque_penalty
                - 0.0 * dof_vel_penalty
                - 0.001 * action_rate_penalty
                + termination_penalty  # 终止条件惩罚
            ),
            # 未到达：正常奖励
            (
                1.5 * tracking_lin_vel    # 提高线速度跟踪权重
                + 0.3 * tracking_ang_vel  # 降低角速度权重
                + approach_reward         # 接近奖励
                + 1.0 * feet_air_time_reward  # go1 风格步态奖励
                - 1.0 * haa_pos_penalty   # 髋关节位置惩罚
                - 0.3 * kfe_pos_penalty   # 膝关节位置惩罚
                - 2.0 * lin_vel_z_penalty
                - 0.05 * ang_vel_xy_penalty
                - 0.0 * orientation_penalty
                - 0.00001 * torque_penalty
                - 0.0 * dof_vel_penalty
                - 0.001 * action_rate_penalty
                + termination_penalty  # 终止条件惩罚
            )
        )

        return reward

    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """
        更新目标位置标记的位置和朝向

        在仿真场景中显示一个绿色箭头标记，指示当前episode的目标位置和朝向。
        标记通过DOF直接控制其世界坐标位置，不参与物理碰撞。

        参数：
            data: 场景数据
            pose_commands: [num_envs, 3] - 目标位姿 (target_x, target_y, target_heading)
        """
        num_envs = data.shape[0]

        # 获取所有环境的dof_pos副本
        all_dof_pos = data.dof_pos.copy()  # [num_envs, num_dof]

        # 为每个环境更新目标标记的位置
        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])       # 目标X坐标
            target_y = float(pose_commands[env_idx, 1])       # 目标Y坐标
            target_z = 0.05  # 平面地形，Z固定为0.05m
            target_yaw = float(pose_commands[env_idx, 2])     # 目标朝向（弧度）

            # 更新target_marker的DOF: [x, y, z, yaw]
            # target_marker有4个DOF：3个平移（slide）+ 1个旋转（hinge）
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x, target_y, target_z, target_yaw
            ]

        # 一次性设置所有环境的dof_pos
        data.set_dof_pos(all_dof_pos, self._model)
        # 调用正向运动学更新body的世界坐标
        self._model.forward_kinematic(data)

    def _compute_projected_gravity(self, quat: np.ndarray) -> np.ndarray:
        """
        计算重力向量在机身坐标系的投影

        将世界坐标系的重力向量 [0, 0, -1] 转换到机身坐标系。
        该值可以用于：
        - 判断机器人的倾斜角度
        - 提供本体感知信息（类似前庭系统）
        - 检测侧翻状态

        参数：
            quat: [num_envs, 4] - 机身姿态四元数 [qx, qy, qz, qw] (Motrix格式)

        返回：
            projected_gravity: [num_envs, 3] - 重力在机身坐标系的投影
                正常站立时约为 [0, 0, -1]
                侧翻时会有较大的XY分量
        """
        # 提取四元数分量 (Motrix格式: [qx, qy, qz, qw])
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # 世界坐标系的重力向量（单位向量，指向下方）
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        vx, vy, vz = gravity_world[0], gravity_world[1], gravity_world[2]

        # 使用四元数旋转公式计算旋转后的向量
        # 公式：v' = q * v * q^(-1)，展开后的矩阵形式
        rx = (1 - 2*(qy*qy + qz*qz)) * vx + 2*(qx*qy - qw*qz) * vy + 2*(qx*qz + qw*qy) * vz
        ry = 2*(qx*qy + qw*qz) * vx + (1 - 2*(qx*qx + qz*qz)) * vy + 2*(qy*qz - qw*qx) * vz
        rz = 2*(qx*qz - qw*qy) * vx + 2*(qy*qz + qw*qx) * vy + (1 - 2*(qx*qx + qy*qy)) * vz

        projected_gravity = np.stack([rx, ry, rz], axis = -1)
        return projected_gravity


    def _compute_terminated(self, state:NpEnvState) -> NpEnvState:
        """
        计算终止条件

        判断哪些环境应该终止当前episode。终止条件包括：
        1. 超时：达到最大episode步数
        2. 速度超限：关节速度过大或出现NaN/Inf
        3. 基座接地：机器人摔倒，基座部分接触地面
        4. 侧翻：机器人倾斜角度超过75°

        参数：
            state: 当前环境状态

        返回：
            更新后的state，其中terminated字段已设置
        """
        data = state.data
        terminated = np.zeros(self._num_envs, dtype = bool)

        # ==================== 1. 超时终止 ====================
        timeout = np.zeros(self._num_envs, dtype=bool)
        if self._cfg.max_episode_steps:
            # 达到最大步数则终止
            timeout = state.info["steps"] >= self._cfg.max_episode_steps
            terminated = np.logical_or(terminated, timeout)

        # ==================== 2. 速度超限终止 ====================
        # 检查DOF速度是否超限（防止数值发散和inf）
        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)  # 每个环境的最大关节速度
        vel_overflow = vel_max > self._cfg.max_dof_vel  # 超过配置的最大速度

        # 极端情况保护：NaN/Inf/超大值
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)
        terminated = np.logical_or(terminated, vel_overflow)
        terminated = np.logical_or(terminated, vel_extreme)

        # ==================== 3. 基座接地终止 ====================
        # 机器人基座接触地面说明摔倒
        cquerys = self._model.get_contact_query(data)
        termination_check = cquerys.is_colliding(self.termination_contact)
        termination_check = termination_check.reshape((self._num_envs, self.num_termination_check))
        base_contact = termination_check.any(axis=1)  # 任何基座部分触地
        terminated = np.logical_or(terminated, base_contact)

        # ==================== 4. 侧翻终止 ====================
        # 倾斜角度超过75°认为侧翻
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        proj_g = self._compute_projected_gravity(root_quat)  # 计算投影重力
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)  # 水平分量
        gz = proj_g[:, 2]  # 垂直分量
        tilt_angle = np.arctan2(gxy, np.abs(gz))  # 倾斜角度
        side_flip_mask = tilt_angle > np.deg2rad(75)  # 超过75度
        terminated = np.logical_or(terminated, side_flip_mask)

        # ==================== 5. 调试统计（可选）====================
        # 统计各类终止原因的数量
        if terminated.any():
            timeout_count = int(timeout.sum())
            vel_count = int((vel_overflow | vel_extreme).sum())
            contact_count = int(base_contact.sum())
            flip_count = int(side_flip_mask.sum())
            total = int(terminated.sum())
            # Every 100 steps, emit termination statistics for debugging.
            if total > 0 and state.info["steps"][0] % 100 == 0:
                logger.debug(
                    "termination total=%s timeout=%s vel=%s contact=%s flip=%s",
                    total,
                    timeout_count,
                    vel_count,
                    contact_count,
                    flip_count,
                )

        return state.replace(terminated = terminated)

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """
        重置环境到初始状态

        该方法在episode开始或终止后调用，执行以下操作：
        1. 随机生成机器人的初始位置
        2. 随机生成目标位姿（相对机器人位置的偏移）
        3. 设置初始DOF状态（位置和速度）
        4. 归一化四元数
        5. 更新可视化标记
        6. 计算初始观测

        参数：
            data: 场景数据
            done: [可选] 指示哪些环境需要重置的布尔数组

        返回：
            obs: [num_envs, 54] - 初始观测
            info: 字典，包含episode信息
        """
        cfg: AnymalCEnvCfg = self._cfg
        num_envs = data.shape[0]

        # ==================== 1. 生成随机初始位置和目标 ====================
        # 机器人的初始位置（在世界坐标系中随机）
        pos_range = cfg.init_state.pos_randomization_range
        robot_init_x = np.random.uniform(
            pos_range[0], pos_range[2],  # x_min, x_max
            num_envs
        )
        robot_init_y = np.random.uniform(
            pos_range[1], pos_range[3],  # y_min, y_max
            num_envs
        )
        robot_init_pos = np.stack([robot_init_x, robot_init_y], axis=1)  # [num_envs, 2]

        # 生成目标位置：相对于机器人初始位置的偏移
        # pose_command_range 定义了目标相对机器人的偏移范围
        target_offset = np.random.uniform(
            low = cfg.commands.pose_command_range[:2],    # [x_min, y_min, yaw_min]前两个
            high = cfg.commands.pose_command_range[3:5],  # [x_max, y_max, yaw_max]前两个
            size = (num_envs, 2)
        )
        target_positions = robot_init_pos + target_offset  # 世界坐标系中的目标位置

        # 生成目标朝向（绝对朝向，在水平方向随机）
        target_headings = np.random.uniform(
            low = cfg.commands.pose_command_range[2],  # yaw_min
            high = cfg.commands.pose_command_range[5], # yaw_max
            size = (num_envs, 1)
        )

        # 组合为完整的位姿命令 [x, y, yaw]
        pose_commands = np.concatenate([target_positions, target_headings],axis = 1)

        # ==================== 2. 设置初始DOF状态 ====================
        # 复制默认的初始状态（所有环境使用相同的默认值）
        init_dof_pos = np.tile(self._init_dof_pos, (*data.shape, 1))
        init_dof_vel = np.tile(self._init_dof_vel, (*data.shape, 1))

        # 创建位置噪声（大部分为0，只设置必要的随机化）
        noise_pos = np.zeros((*data.shape, self._num_dof_pos), dtype=np.float32)

        # target_marker (DOF 0-3): 不添加噪声，会在_update_target_marker中设置

        # base的位置 (DOF 4-6): 设置为前面生成的随机初始位置
        noise_pos[:, 4] = robot_init_x - cfg.init_state.pos[0]  # X轴偏移
        noise_pos[:, 5] = robot_init_y - cfg.init_state.pos[1]  # Y轴偏移
        # Z轴不添加噪声，保持固定高度，避免坠落感
        # base的四元数 (DOF 7-10): 不添加噪声，保持为单位四元数

        # 关节角度(DOF 11:)不添加噪声，保证初始站立稳定
        # noise_pos[:, 11:] = 0.0  # 已经初始化为0

        # 所有速度都设为0，确保完全静止
        noise_vel = np.zeros((*data.shape, self._num_dof_vel), dtype=np.float32)

        dof_pos = init_dof_pos + noise_pos
        dof_vel = init_dof_vel + noise_vel

        # 归一化base的四元数（DOF 7-10）
        # 新的DOF结构：target_marker占0-3, base_pos占4-6, base_quat占7-10
        for env_idx in range(num_envs):
            quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]  # [qx, qy, qz, qw]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:  # 避免除以零
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = quat / quat_norm
            else:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 默认单位四元数

            # 归一化箭头的四元数（如果箭头body存在）
            if self._robot_arrow_body is not None:
                # robot_heading_arrow的四元数（DOF 25-28: qx, qy, qz, qw）
                robot_arrow_quat = dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end]
                quat_norm = np.linalg.norm(robot_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = robot_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

                # desired_heading_arrow的四元数（DOF 32-35: qx, qy, qz, qw）
                desired_arrow_quat = dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end]
                quat_norm = np.linalg.norm(desired_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = desired_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        # 更新目标位置标记
        self._update_target_marker(data, pose_commands)

        # 获取根节点状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)

        # 关节状态（腿部关节）
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        # 获取传感器数据
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # 计算速度命令（与update_state一致）
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]

        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)  # [num_envs]

        # 位置阈值：0.1米内认为到达
        position_threshold = 0.1
        reached_position = distance_to_target < position_threshold  # [num_envs]

        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)  # 到达后速度为0

        # 实际线速度 XY
        base_lin_vel_xy = base_lin_vel[:, :2]

        # 更新箭头可视化（不影响物理）
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)

        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)

        # 朝向阈值：15度内认为到达
        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold  # [num_envs]

        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        reached_all = np.logical_and(reached_position, reached_heading)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)  # 到达后觗速度为0
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)  # 到达后速度为0

        # 确保 desired_yaw_rate 是1维数组
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()

        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )

        # 归一化观测（与update_state一致）
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)

        # 计算任务相关观测（与update_state一致）
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)

        # 计算是否达到zero_ang标准
        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)

        obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3
                position_error_normalized,  # 2
                heading_error_normalized[:, np.newaxis],  # 1
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
            ],
            axis=-1,
        )
        assert obs.shape == (num_envs, 54)


        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "min_distance": distance_to_target.copy(),  # 初始化最小距离
            # go1 风格的 feet_air_time 追踪
            "feet_air_time": np.zeros((num_envs, self.num_foot_check), dtype=np.float32),
            "contacts": np.zeros((num_envs, self.num_foot_check), dtype=bool),
        }

        return obs, info


AnymalCNavTask = AnymalCEnv


@registry.env("anymal_c_navigation_rough", "np")
class AnymalCRoughEnv(AnymalCEnv):
    """ANYmal C 崎岖地形导航环境

    基于平面导航环境扩展，增加：
    - 地形高度网格初始化
    - 边界检查和终止条件
    - 渐进式训练级别
    """
    _cfg: AnymalCRoughEnvCfg

    def __init__(self, cfg: AnymalCRoughEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs)
        # 地形相关初始化
        self._init_terrain_grid()
        self._training_level = 0
        self._height_counter = 0

    def _init_terrain_grid(self):
        """初始化崎岖地形网格

        参考 go1-rough-terrain-walk 的成功实现：
        使用预计算的离散高度值，而不是动态查询 heightmap。
        这样可以确保机器人总是被放置在正确的地形高度上。

        heightfield 参数 (scene_rough.xml):
        - size="20 20 2.5 0.1" -> 40m x 40m, 最大高度2.5m
        - pos="0 0 -1" -> 基准位置Z=-1
        - 地形高度范围: -1m 到 +1.5m
        """
        cfg = self._cfg

        # heightfield 参数
        self._hfield_size_x = 20.0  # X方向半尺寸 (米)
        self._hfield_size_y = 20.0  # Y方向半尺寸 (米)
        self._hfield_max_height = 2.5  # 最大高度 (米)
        self._hfield_base_z = -1.0  # heightfield 基准 Z 位置

        # 加载 heightmap 图像用于查询
        heightmap_path = os.path.dirname(__file__) + "/xmls/assets/heightmap.png"
        try:
            from PIL import Image
            heightmap_img = Image.open(heightmap_path).convert('L')  # 转为灰度
            self._heightmap = np.array(heightmap_img, dtype=np.float32) / 255.0  # 归一化到 [0, 1]
            self._heightmap_shape = self._heightmap.shape  # (height, width)
        except Exception as e:
            logger.warning("Failed to load heightmap %s: %s", heightmap_path, e)
            self._heightmap = np.zeros((257, 257), dtype=np.float32)
            self._heightmap_shape = (257, 257)

        # 机器人站立高度（从脚底到基座）
        self._robot_standing_height = 0.56

        # ==================== 参考 go1 的预计算方式 ====================
        # go1 使用预定义的离散高度值，这里我们也采用类似的策略
        # 根据 heightmap 分析，主要的高度区域有：
        # - 低地形区 (Z ≈ -0.2 ~ 0.2)
        # - 中地形区 (Z ≈ 0.4 ~ 0.7)
        # - 高地形区 (Z ≈ 1.0 ~ 1.3)

        # 预定义三个安全的高度级别（世界坐标）
        # 这些值是基于 heightmap 分析确定的
        self._height_list = np.array([0.5, 0.8, 1.1], dtype=np.float32)

        # 构建 5x5 网格的位置和高度
        # 参考 go1: offset_h 定义每个网格位置的高度索引
        grid_size = cfg.terrain_config.grid_size
        cell_size = cfg.terrain_config.cell_size

        # 预计算每个网格位置的高度（采样 heightmap）
        offset = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i - grid_size // 2) * cell_size
                y = (j - grid_size // 2) * cell_size
                # 查询该位置的地形高度
                terrain_z = self._get_terrain_height(np.array([x]), np.array([y]))[0]
                # 计算机器人应该在的 Z 坐标（地形高度 + 站立高度 + 余量）
                robot_z = terrain_z + self._robot_standing_height + 0.2
                offset.append([x, y, robot_z])

        self._offset_list = np.array(offset, dtype=np.float32)

        # 设置初始 Z 位置（原点处）
        origin_terrain_height = self._get_terrain_height(np.array([0.0]), np.array([0.0]))[0]
        initial_z = origin_terrain_height + self._robot_standing_height + 0.2
        self._init_dof_pos[6] = initial_z  # DOF 6 是 base 的 Z 位置

    def _get_terrain_height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """根据世界坐标 (x, y) 查询地形高度

        参数:
            x: [N] 世界坐标 X
            y: [N] 世界坐标 Y

        返回:
            heights: [N] 地形表面的世界 Z 坐标

        注意:
            MuJoCo heightfield 的坐标映射：
            - 图像 row=0 (顶部) 对应世界 Y = +size_y
            - 图像 row=max (底部) 对应世界 Y = -size_y
            - 图像 col=0 (左侧) 对应世界 X = -size_x
            - 图像 col=max (右侧) 对应世界 X = +size_x
            因此 Y 轴需要翻转。
        """
        # 将世界坐标转换为 heightmap 像素坐标
        # 世界坐标范围: [-20, 20] -> 像素坐标范围: [0, image_size-1]
        img_h, img_w = self._heightmap_shape

        # 归一化到 [0, 1]
        norm_x = (x / self._hfield_size_x + 1.0) / 2.0  # [-20, 20] -> [0, 1]
        norm_y = (y / self._hfield_size_y + 1.0) / 2.0  # [-20, 20] -> [0, 1]

        # 转换为像素坐标
        # X 轴: 直接映射
        pixel_x = np.clip((norm_x * (img_w - 1)).astype(np.int32), 0, img_w - 1)
        # Y 轴: 需要翻转，因为 MuJoCo heightfield 的 row=0 对应 Y=+size_y
        pixel_y = np.clip(((1.0 - norm_y) * (img_h - 1)).astype(np.int32), 0, img_h - 1)

        # 从 heightmap 采样高度值 (0-1)
        height_values = self._heightmap[pixel_y, pixel_x]

        # 转换为世界 Z 坐标
        # world_z = base_z + height_value * max_height
        world_z = self._hfield_base_z + height_values * self._hfield_max_height

        return world_z

    def _get_terrain_slope(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算指定位置的地形坡度

        通过采样周围点的高度来估算坡度。

        参数:
            x: [N] 世界坐标 X
            y: [N] 世界坐标 Y

        返回:
            slopes: [N] 坡度（弧度），0表示平坦，π/2表示垂直
        """
        # 采样间距（米）
        delta = 0.1

        # 采样周围4个点
        h_center = self._get_terrain_height(x, y)
        h_px = self._get_terrain_height(x + delta, y)
        h_mx = self._get_terrain_height(x - delta, y)
        h_py = self._get_terrain_height(x, y + delta)
        h_my = self._get_terrain_height(x, y - delta)

        # 计算 X 和 Y 方向的梯度
        grad_x = (h_px - h_mx) / (2 * delta)
        grad_y = (h_py - h_my) / (2 * delta)

        # 坡度 = arctan(sqrt(grad_x^2 + grad_y^2))
        slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2))

        return slope

    def _border_check(self, data, info):
        """检查机器人是否到达地形边界"""
        border_size = self._cfg.terrain_config.border_size  # 从配置读取地形边界半径
        position = self._body.get_position(data)
        is_out = (np.square(position[:, :2]) > border_size**2).any(axis=1)
        # 将越界机器人的目标设为原点
        info["pose_commands"][is_out] = [0, 0, 0]

    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """
        更新目标位置标记的位置和朝向（崎岖地形版本）

        根据地形高度动态设置目标标记的 Z 坐标，确保标记显示在地形表面上方。

        参数：
            data: 场景数据
            pose_commands: [num_envs, 3] - 目标位姿 (target_x, target_y, target_heading)
        """
        num_envs = data.shape[0]

        # 获取所有环境的dof_pos副本
        all_dof_pos = data.dof_pos.copy()  # [num_envs, num_dof]

        # 获取所有目标位置的 X、Y 坐标
        target_x_arr = pose_commands[:, 0].astype(np.float32)
        target_y_arr = pose_commands[:, 1].astype(np.float32)

        # 批量查询地形高度
        terrain_heights = self._get_terrain_height(target_x_arr, target_y_arr)

        # 为每个环境更新目标标记的位置
        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])
            target_y = float(pose_commands[env_idx, 1])
            target_z = float(terrain_heights[env_idx]) + 0.1  # 地形高度 + 0.1m 余量
            target_yaw = float(pose_commands[env_idx, 2])

            # 更新target_marker的DOF: [x, y, z, yaw]
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x, target_y, target_z, target_yaw
            ]

        # 一次性设置所有环境的dof_pos
        data.set_dof_pos(all_dof_pos, self._model)
        # 调用正向运动学更新body的世界坐标
        self._model.forward_kinematic(data)

    def update_state(self, state: NpEnvState):
        """重写状态更新，添加边界检查"""
        # 先进行边界检查
        self._border_check(state.data, state.info)
        # 调用父类的状态更新
        return super().update_state(state)

    def _compute_terminated(self, state: NpEnvState) -> NpEnvState:
        """添加边界终止条件"""
        state = super()._compute_terminated(state)

        # 检查是否超出边界
        position = self._body.get_position(state.data)
        border_size = self._cfg.terrain_config.border_size  # 从配置读取地形边界半径
        out_of_bounds = (np.square(position[:, :2]) > border_size**2).any(axis=1)
        state.terminated = np.logical_or(state.terminated, out_of_bounds)

        return state

    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray) -> np.ndarray:
        """计算奖励函数（崎岖地形版本）

        继承基类奖励函数，并添加崎岖地形特有的奖励项：

        1. 地形稳定性奖励 (terrain_stability_reward):
           - 使用高斯核函数，Z轴速度越小奖励越高
           - 相比阶跃函数更平滑，有利于策略学习

        2. 姿态稳定性惩罚 (orientation_penalty):
           - 惩罚机身倾斜，鼓励在崎岖地形上保持平衡
           - 在崎岖地形中比平面地形更重要

        3. 足部接触奖励 (foot_contact_reward):
           - 鼓励稳定的四足着地
           - 提高在不平地面上的稳定性
        """
        # 调用基类的奖励计算
        reward = super()._compute_reward(data, info, velocity_commands)

        # ==================== 1. 地形稳定性奖励 ====================
        # 使用高斯核函数替代阶跃函数，更平滑且区分度更高
        # sigma=0.1 使得 lin_vel_z 在 ±0.3 m/s 范围内获得显著奖励
        base_lin_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        lin_vel_z = base_lin_vel[:, 2]  # Z轴线速度
        terrain_stability_reward = 0.5 * np.exp(-np.square(lin_vel_z) / 0.1)

        # ==================== 2. 姿态稳定性惩罚 ====================
        # 在崎岖地形中，保持机身水平更加重要
        # 惩罚 projected_gravity 的 XY 分量（理想状态下应为0）
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        proj_g = self._compute_projected_gravity(root_quat)
        # proj_g 理想值为 [0, 0, -1]，XY分量越大说明倾斜越严重
        orientation_penalty = np.square(proj_g[:, 0]) + np.square(proj_g[:, 1])

        # ==================== 3. 足部接触奖励 ====================
        # 鼓励稳定的足部接触，更多脚着地 = 更稳定
        cquerys = self._model.get_contact_query(data)
        foot_contact = cquerys.is_colliding(self.foot_contact_check)
        foot_contact = foot_contact.reshape((data.shape[0], self.num_foot_check))
        # 归一化到 [0, 1]，4脚全部着地时为1
        foot_contact_reward = 0.2 * foot_contact.sum(axis=1).astype(np.float32) / max(self.num_foot_check, 1)

        # ==================== 汇总崎岖地形奖励 ====================
        rough_terrain_reward = (
            terrain_stability_reward      # 地形稳定性奖励 (最大0.5)
            - 0.15 * orientation_penalty  # 姿态稳定性惩罚
            + foot_contact_reward         # 足部接触奖励 (最大0.2)
        )

        return reward + rough_terrain_reward

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """重写重置，支持渐进式地形放置"""
        cfg: AnymalCRoughEnvCfg = self._cfg
        num_envs = data.shape[0]

        # ==================== 1. 生成随机初始位置和目标 ====================
        # 机器人的初始位置（在世界坐标系中随机）
        pos_range = cfg.init_state.pos_randomization_range
        robot_init_x = np.random.uniform(
            pos_range[0], pos_range[2],  # x_min, x_max
            num_envs
        )
        robot_init_y = np.random.uniform(
            pos_range[1], pos_range[3],  # y_min, y_max
            num_envs
        )
        robot_init_pos = np.stack([robot_init_x, robot_init_y], axis=1)  # [num_envs, 2]

        # 获取机器人初始位置的地形高度
        robot_terrain_heights = self._get_terrain_height(robot_init_x, robot_init_y)

        # 生成目标位置，检查高度差和坡度是否可接受
        max_height_diff = getattr(cfg.commands, 'max_height_diff', 0.5)
        max_target_slope = getattr(cfg.commands, 'max_target_slope', 0.2)
        max_resample_attempts = getattr(cfg.commands, 'max_resample_attempts', 20)

        target_positions = np.zeros((num_envs, 2), dtype=np.float32)
        for env_idx in range(num_envs):
            best_target = None
            best_score = float('inf')  # 越小越好

            for attempt in range(max_resample_attempts):
                # 生成目标偏移
                target_offset = np.random.uniform(
                    low=cfg.commands.pose_command_range[:2],
                    high=cfg.commands.pose_command_range[3:5],
                )
                target_pos = robot_init_pos[env_idx] + target_offset

                # 检查目标位置的地形高度
                target_height = self._get_terrain_height(
                    np.array([target_pos[0]]),
                    np.array([target_pos[1]])
                )[0]

                # 计算高度差
                height_diff = abs(target_height - robot_terrain_heights[env_idx])

                # 检查目标点的坡度
                target_slope = self._get_terrain_slope(
                    np.array([target_pos[0]]),
                    np.array([target_pos[1]])
                )[0]

                # 综合评分：高度差和坡度都要满足条件
                height_ok = height_diff <= max_height_diff
                slope_ok = target_slope <= max_target_slope

                if height_ok and slope_ok:
                    # 找到合适的目标点
                    target_positions[env_idx] = target_pos
                    break
                else:
                    # 记录最佳候选（如果所有尝试都失败，使用最接近条件的）
                    score = height_diff / max_height_diff + target_slope / max_target_slope
                    if score < best_score:
                        best_score = score
                        best_target = target_pos
            else:
                # 超过最大尝试次数，使用最佳候选或出生点附近的安全位置
                if best_target is not None and best_score < 3.0:
                    # 使用最接近条件的目标点
                    target_positions[env_idx] = best_target
                else:
                    # 使用出生点附近的安全位置（小范围偏移）
                    safe_offset = np.random.uniform(-0.5, 0.5, size=2)
                    target_positions[env_idx] = robot_init_pos[env_idx] + safe_offset

        # 生成目标朝向（绝对朝向，在水平方向随机）
        target_headings = np.random.uniform(
            low=cfg.commands.pose_command_range[2],
            high=cfg.commands.pose_command_range[5],
            size=(num_envs, 1)
        )

        # 组合为完整的位姿命令 [x, y, yaw]
        pose_commands = np.concatenate([target_positions, target_headings], axis=1)

        # ==================== 2. 设置初始DOF状态 ====================
        # 复制默认的初始状态
        init_dof_pos = np.tile(self._init_dof_pos, (*data.shape, 1))
        init_dof_vel = np.tile(self._init_dof_vel, (*data.shape, 1))

        # 渐进式训练：根据训练级别决定放置策略
        # 参考 go1-rough-terrain-walk 的实现
        if self._training_level >= 1:
            # 高级阶段：在预计算的网格位置上循环放置
            num_period = cfg.terrain_config.grid_size ** 2  # grid_size x grid_size 网格
            idx = np.arange(self._height_counter, self._height_counter + num_envs) % num_period
            self._height_counter = (self._height_counter + num_envs) % num_period
            # 直接使用预计算的 [x, y, z] 位置
            spawn_x = self._offset_list[idx, 0]
            spawn_y = self._offset_list[idx, 1]
            spawn_z = self._offset_list[idx, 2]
        else:
            # 初级阶段：在随机位置放置，动态计算高度
            spawn_x = robot_init_x
            spawn_y = robot_init_y
            # 查询地形高度，计算正确的生成 Z 坐标
            terrain_heights = self._get_terrain_height(spawn_x, spawn_y)
            # 增加余量，确保机器人不会穿透地形
            spawn_z = terrain_heights + self._robot_standing_height + 0.2

        # 设置 base 位置 [x, y, z] (DOF 4-6)
        init_dof_pos[:, 4] = spawn_x
        init_dof_pos[:, 5] = spawn_y
        init_dof_pos[:, 6] = spawn_z

        # 归一化base的四元数（DOF 7-10）
        for env_idx in range(num_envs):
            quat = init_dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:
                init_dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = quat / quat_norm
            else:
                init_dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            # 归一化箭头的四元数（如果箭头body存在）
            if self._robot_arrow_body is not None:
                robot_arrow_quat = init_dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end]
                quat_norm = np.linalg.norm(robot_arrow_quat)
                if quat_norm > 1e-6:
                    init_dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = robot_arrow_quat / quat_norm
                else:
                    init_dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

                desired_arrow_quat = init_dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end]
                quat_norm = np.linalg.norm(desired_arrow_quat)
                if quat_norm > 1e-6:
                    init_dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = desired_arrow_quat / quat_norm
                else:
                    init_dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        data.reset(self._model)
        data.set_dof_vel(init_dof_vel)
        data.set_dof_pos(init_dof_pos, self._model)
        self._model.forward_kinematic(data)

        # 更新目标位置标记
        self._update_target_marker(data, pose_commands)

        # 获取根节点状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)

        # 关节状态（腿部关节）
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        # 获取传感器数据
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # 计算速度命令（与update_state一致）
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]

        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        position_threshold = 0.1
        reached_position = distance_to_target < position_threshold

        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)

        # 实际线速度 XY
        base_lin_vel_xy = base_lin_vel[:, :2]

        # 更新箭头可视化（不影响物理）
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)

        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)

        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold

        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        reached_all = np.logical_and(reached_position, reached_heading)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)

        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()

        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )

        # 归一化观测
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)

        # 计算任务相关观测
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)

        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)

        obs = np.concatenate(
            [
                noisy_linvel,
                noisy_gyro,
                projected_gravity,
                noisy_joint_angle,
                noisy_joint_vel,
                last_actions,
                command_normalized,
                position_error_normalized,
                heading_error_normalized[:, np.newaxis],
                distance_normalized[:, np.newaxis],
                reached_flag[:, np.newaxis],
                stop_ready_flag[:, np.newaxis],
            ],
            axis=-1,
        )
        assert obs.shape == (num_envs, 54)

        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "min_distance": distance_to_target.copy(),
            # go1 风格的 feet_air_time 追踪
            "feet_air_time": np.zeros((num_envs, self.num_foot_check), dtype=np.float32),
            "contacts": np.zeros((num_envs, self.num_foot_check), dtype=bool),
        }

        return obs, info

    def _update_training_level(self, reward: np.ndarray):
        """根据平均奖励更新训练级别"""
        average_reward = np.average(reward)
        if 0.9 < average_reward and self._training_level == 0:
            self._training_level = 1
            logger.info("AnymalCRoughEnv training level upgraded to %s", self._training_level)
