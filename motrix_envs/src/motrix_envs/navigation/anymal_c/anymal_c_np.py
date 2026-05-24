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

"""Anymal C导航环境实现。

该模块实现了Anymal C四足机器人在平坦地形上的导航任务环境。
主要包含：
- AnymalCNavTask：完整的环境类实现
- quat_rotate_inverse：四元数旋转辅助函数
- 各种奖励函数：位置追踪、速度追踪、动作平滑等
- 观测计算：融合传感器数据和导航信息
- 状态管理：重置、步进、奖励计算等
"""

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.navigation.anymal_c.cfg import AnymalCNavEnvCfg
from motrix_envs.navigation.anymal_c.reward_utils import apply_termination_penalty
from motrix_envs.np.env import NpEnv, NpEnvState


def quat_rotate_inverse(quats, v):
    """将固定向量v通过四元数列表进行反向旋转（向量世界坐标系→机器人体坐标系）。

    该函数使用向量化方法一次性处理多个四元数，用于将世界坐标系下的向量
    旋转到机器人身体坐标系。常用于将重力向量/速度向量变换到机器人局部坐标。

    数学原理：
    对于四元数 q = [x, y, z, w]，其共轭为 q* = [-x, -y, -z, w]
    向量v在机器人坐标系下的表示为：v' = q* ∘ v ∘ q
    其中∘表示四元数乘积

    Args:
        quats (np.ndarray): 四元数数组，形状为 (N, 4)，格式为 [x, y, z, w]
                           代表世界坐标系到机器人坐标系的旋转
        v (np.ndarray): 固定向量，形状为 (3,)，在世界坐标系中的坐标

    Returns:
        np.ndarray: 旋转后的向量数组，形状为 (N, 3)，在机器人坐标系中的坐标

    使用示例：
        >>> quats = np.array([[0, 0, 0, 1], [0, 0, 0.7071, 0.7071]])  # 两个四元数
        >>> gravity = np.array([0, 0, -1])  # 重力向量
        >>> rotated = quat_rotate_inverse(quats, gravity)  # 获得重力在机器人坐标的投影
    """
    # =============== 步骤1：提取四元数的虚部和实部 ===============
    # 实部(标量)：w分量，形状 (N,)
    w = quats[:, -1]
    # 虚部(向量)：x, y, z分量，形状 (N, 3)
    im = quats[:, :3]

    # =============== 步骤2：计算叉积（向量叉积） ===============
    # cross(im, v)：虚部与向量的叉积，形状 (N, 3)
    cross_im_v = np.cross(im, v)

    # =============== 步骤3：四元数旋转公式中间项 ===============
    # term1 = w * cross(im, v)：实部系数项
    term1 = w[:, np.newaxis] * cross_im_v
    # term2 = cross(im, cross(im, v))：二阶交叉项
    term2 = np.cross(im, cross_im_v)

    # =============== 步骤4：应用旋转公式 ===============
    # 四元数旋转公式：v_rot = v + 2 * (term1 + term2)
    # 这是共轭四元数旋转的向量形式，用于计算向量q* v q的结果
    v_rotated = v + 2 * (term1 + term2)

    return v_rotated


@registry.env("anymal-c-flat-terrain-nav", sim_backend="np")
class AnymalCNavTask(NpEnv):
    """Anymal C平坦地形导航任务环境。

    该类实现了一个完整的强化学习环境，用于训练Anymal C四足机器人进行导航。
    环境特点：
    - 基于MotrixSim物理引擎的NumPy后端实现
    - 支持并行化多个环境实例
    - 包含完整的奖励函数体系（导航+运动控制）
    - 支持接触检测和回合终止条件
    - 融合多个传感器的观测向量

    环境注册名称: "anymal-c-flat-terrain-nav"
    模拟后端: NumPy (sim_backend="np")

    属性：
        _init_dof_pos: 初始关节位置 (19,)
        _init_dof_vel: 初始关节速度 (18,)
        _body: 机器人身体对象引用
        _num_action: 动作维度 = 12
        _num_observation: 观测维度 = 48
        _num_dof_pos: 位置自由度数 = 19
        _num_dof_vel: 速度自由度数 = 18
    """

    # =============== 类型注解 ===============
    _init_dof_pos: np.ndarray  # 初始关节位置
    _init_dof_vel: np.ndarray  # 初始关节速度

    def __init__(self, cfg: AnymalCNavEnvCfg, num_envs=1):
        """初始化Anymal C导航环境。

        该初始化方法实现了以下关键步骤：
        1. 获取关键 body、joint、actuator 信息
        2. 定义动作和观测空间
        3. 初始化默认状态
        4. 设置缓冲区

        Args:
            cfg: 环境配置对象 (AnymalCNavEnvCfg)
            num_envs: 并行环境数量，默认为1
        """
        # 调用父类初始化，完成基础配置和模型加载
        super().__init__(cfg, num_envs)

        # =============== 步骤2: 定义动作和观测空间 ===============
        # 初始化动作空间：12个关节的位置目标 [-1, 1]范围
        self._init_action_space()
        # 初始化观测空间：48维度的观测向量
        self._init_obs_space()

        # =============== 步骤1: 获取关键body、joint、actuator ===============
        # 从模型中获取机器人身体(base)的引用，用于查询位置/姿态/接触等信息
        self._body = self._model.get_body(self.cfg.asset_cfg.body_name)

        # 获取动作和观测的维度信息
        self._num_action = self._action_space.shape[0]  # 12个关节
        self._num_observation = self._observation_space.shape[0]  # 48维观测

        # 从模型获取关节(DOF)信息
        self._num_dof_pos = self._model.num_dof_pos  # 自由度位置维度(7+12=19)
        self._num_dof_vel = self._model.num_dof_vel  # 自由度速度维度(6+12=18)

        # =============== 步骤3: 初始化默认状态 ===============
        # 初始化所有关节速度为0 (shape: [num_dof_vel])
        self._init_dof_vel = np.zeros(
            (self._num_dof_vel,),
            dtype=np.float32,
        )

        # 计算初始关节位置（包括base位置/姿态和关节角度）
        # 基于配置中的init_state设置关节角度
        self._init_dof_pos = self._model.compute_init_dof_pos()

        # =============== 步骤4: 设置缓冲区 ===============
        # 初始化各种缓冲区：
        # - reset_buf: 标记需要重置的环境
        # - kps/kds: PD控制器参数
        # - gravity_vec: 重力向量
        # - default_angles: 默认关节角度
        # - hip_indices: 髋关节索引列表
        # - ground/foot/termination_contact: 接触检测相关
        # - feet_air_time: 脚悬空时间追踪
        self._init_buffer()

    def _init_obs_space(self):
        """初始化观测空间。

        观测向量是一个48维的向量，包含以下部分：

        观测组成（总计48维）：
        ┌─────────────────────────────────────────────┐
        │ 1. Base线速度（局部坐标）: 3维              │
        │    - vx, vy, vz（机器人坐标系）            │
        ├─────────────────────────────────────────────┤
        │ 2. Base角速度（陀螺仪）: 3维                │
        │    - wx, wy, wz（机器人坐标系）            │
        ├─────────────────────────────────────────────┤
        │ 3. 重力向量投影（局部坐标）: 3维            │
        │    - 世界重力向量在机器人坐标系的投影      │
        ├─────────────────────────────────────────────┤
        │ 4. 关节角度偏差: 12维                      │
        │    - 相对于默认站立姿态的偏差             │
        ├─────────────────────────────────────────────┤
        │ 5. 关节速度: 12维                          │
        │    - 12个关节的速度 [rad/s]               │
        ├─────────────────────────────────────────────┤
        │ 6. 上一步动作: 12维                        │
        │    - 用于连续性约束和动作变化奖励         │
        ├─────────────────────────────────────────────┤
        │ 7. 导航命令: 3维                           │
        │    - 相对位置x (机器人坐标) [m]           │
        │    - 相对位置y (机器人坐标) [m]           │
        │    - 相对朝向 (偏差) [rad]                │
        └─────────────────────────────────────────────┘

        所有观测值都进行了归一化处理，范围约为[-1, 1]。
        """
        model = self.model
        # base线速度 + 陀螺仪 + 关节速度 = 6 + 12 = 18维
        num_dof_vel = model.num_dof_vel
        # 关节角度 = 总DOF - base DOF = 19 - 7 = 12维
        num_joint_angle = model.num_dof_pos - 7
        # 重力向量投影 = 3维
        num_gravity = 3
        # 关节数 = 12维
        num_actions = model.num_actuators
        # 导航命令 = 相对x, 相对y, 相对朝向 = 3维
        num_command = 3

        # 计算总观测维度
        num_obs = num_dof_vel + num_joint_angle + num_gravity + num_actions + num_command
        assert num_obs == 48, f"Expected 48, got {num_obs}"

        # 创建观测空间：无界连续空间
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

    def _init_action_space(self):
        """初始化动作空间。

        动作空间是一个12维的连续空间，代表12个关节的位置目标。
        每个关节都由一个PD控制器控制，将动作（位置目标）转换为关节扭矩。

        动作构成：
        - 左前腿(LF): HAA, HFE, KFE (3个)
        - 右前腿(RF): HAA, HFE, KFE (3个)
        - 左后腿(LH): HAA, HFE, KFE (3个)
        - 右后腿(RH): HAA, HFE, KFE (3个)
        总计: 12个关节

        动作范围：
        - 上界(lower): 从执行器控制限制的最小值读取
        - 下界(upper): 从执行器控制限制的最大值读取
        - 典型范围: [-1, 1] (由动作缩放系数 action_scale 进一步调整)
        """
        model = self.model
        self._action_space = gym.spaces.Box(
            # 下界：从模型的执行器控制限制读取
            np.array(model.actuator_ctrl_limits[0, :]),
            # 上界：从模型的执行器控制限制读取
            np.array(model.actuator_ctrl_limits[1, :]),
            # 形状：(12,) 对应12个关节
            (model.num_actuators,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        """返回动作空间。

        动作空间是一个12维的连续盒子空间，代表12个关节的位置目标。
        范围由执行器控制限制定义，通常为[-1, 1]。

        Returns:
            gym.spaces.Box: 动作空间，形状 (12,)，范围 [-1, 1]
        """
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        """返回观测空间。

        观测空间是一个48维的连续盒子空间，范围无界。
        包含线速度、角速度、重力投影、关节角度、关节速度、
        上一步动作和导航命令。

        Returns:
            gym.spaces.Box: 观测空间，形状 (48,)，范围 [-∞, ∞]
        """
        return self._observation_space

    def get_dof_pos(self, data: mtx.SceneModel):
        """获取关节位置（DOF位置）。

        从body对象获取所有关节的当前位置（角度）。
        不包括base的位置/朝向（那些在pose中）。

        Args:
            data (mtx.SceneData): 仿真数据对象

        Returns:
            np.ndarray: 关节位置数组，形状 (num_envs, 12)，单位 [rad]
        """
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneModel):
        """获取关节速度（DOF速度）。

        从body对象获取所有关节的当前速度（角速度）。
        不包括base的线速度/角速度（那些在陀螺仪和线速度传感器中）。

        Args:
            data (mtx.SceneData): 仿真数据对象

        Returns:
            np.ndarray: 关节速度数组，形状 (num_envs, 12)，单位 [rad/s]
        """
        return self._body.get_joint_dof_vel(data)

    def _init_buffer(self):
        """初始化缓冲区和索引。

        该方法初始化了环境所有的缓冲区和辅助索引，包括：
        1. 环境重置标记
        2. PD控制器参数
        3. 物理常量（重力）
        4. 观测/命令归一化系数
        5. 关节相关信息（默认角度、髋关节索引）
        6. 接触检测信息（地面、脚、终止接触）

        这些缓冲区在整个环境生命周期内保持不变。
        """
        cfg = self._cfg
        assert isinstance(cfg, AnymalCNavEnvCfg)

        # =============== 环境重置标记 ===============
        # 初始化所有环境为需要重置状态
        # 形状: (num_envs,)
        self.reset_buf = np.ones(self._num_envs, dtype=np.bool_)

        # =============== PD控制器参数 ===============
        # 刚度系数 Kp：12个关节都使用相同的刚度值
        # 用于公式：τ = Kp * (θ_target - θ_actual) - Kd * ω_actual
        self.kps = np.ones(self._num_action, dtype=np.float32) * cfg.control_cfg.stiffness

        # 阻尼系数 Kd：12个关节都使用相同的阻尼值
        self.kds = np.ones(self._num_action, dtype=np.float32) * cfg.control_cfg.damping

        # =============== 物理常量 ===============
        # 重力加速度向量（世界坐标系中向下为负）
        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)

        # =============== 命令归一化系数 ===============
        # 用于将原始命令缩放到合适范围
        # 包含：[lin_vel_x, lin_vel_y, ang_vel_z]的归一化系数
        self.commands_scale = np.array(
            [
                cfg.normalization_cfg.lin_vel,    # x方向线速度归一化
                cfg.normalization_cfg.lin_vel,    # y方向线速度归一化
                cfg.normalization_cfg.ang_vel,    # z方向角速度归一化
            ],
            dtype=np.float32,
        )

        # =============== 初始化默认关节角度和髋关节索引 ===============
        # 默认关节角度数组：12维
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        # 髋关节的索引列表，用于某些奖励计算
        self.hip_indices = []

        # 遍历所有执行器，建立名称和默认角度的对应关系
        for i in range(self._model.num_actuators):
            # 从配置中查找对应的默认角度
            for name in cfg.init_state_cfg.default_joint_angles.keys():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = cfg.init_state_cfg.default_joint_angles[name]
            # 标记所有髋关节(HAA - Hip Abduction/Adduction)
            if "HAA" in self._model.actuator_names[i]:
                self.hip_indices.append(i)

        # 将默认关节角度写入初始状态
        # _init_dof_pos = [base_pos(3) + base_quat(4) + joint_pos(12)]
        #                = [0:7] + [7:19]
        self._init_dof_pos[-self._num_action:] = self.default_angles

        # =============== 接触检测设置：回合终止条件 ===============
        # 获取地面几何体的索引
        self.ground = self._model.get_geom_index(cfg.asset_cfg.ground)
        # 回合终止接触对列表（触发则结束）
        self.termination_contact = None
        self.foot = []

        # 设置哪些部位与地面接触会导致回合终止（如身体下陷=摔倒）
        for name in cfg.asset_cfg.terminate_after_contacts_on:
            # 首先尝试查找带_collision后缀的几何体
            geom_idx = self._model.get_geom_index(f"{name}_collision")
            if geom_idx is None:
                # 否则直接查找该名称的几何体
                geom_idx = self._model.get_geom_index(name)

            if geom_idx is not None:
                # 添加接触对 [几何体索引, 地面索引]
                if self.termination_contact is None:
                    self.termination_contact = np.array([[geom_idx, self.ground]], dtype=np.uint32)
                else:
                    self.termination_contact = np.append(
                        self.termination_contact,
                        np.array([[geom_idx, self.ground]], dtype=np.uint32),
                        axis=0,
                    )

        # 统计需要检查的终止接触对数量
        if self.termination_contact is not None:
            self.num_check = self.termination_contact.shape[0]
        else:
            self.num_check = 0
            self.termination_contact = np.array([], dtype=np.uint32).reshape(0, 2)

        # =============== 接触检测设置：脚部接触 ===============
        # 查找所有脚部几何体，用于检测脚部是否着地
        self.foot = None
        for i in self._model.geom_names:
            # 查找名称中包含"foot"的几何体
            if i is not None and cfg.asset_cfg.foot_name in i:
                geom_idx = self._model.get_geom_index(i)
                if geom_idx is not None:
                    # 添加脚与地面的接触对 [脚索引, 地面索引]
                    if self.foot is None:
                        self.foot = np.array([[geom_idx, self.ground]], dtype=np.uint32)
                    else:
                        self.foot = np.append(
                            self.foot,
                            np.array([[geom_idx, self.ground]], dtype=np.uint32),
                            axis=0,
                        )

        # 统计脚部接触对数量和构建脚部检测列表
        if self.foot is not None:
            self.foot_check_num = self.foot.shape[0]
            self.foot_check = self.foot
        else:
            self.foot_check_num = 0
            self.foot_check = np.array([], dtype=np.uint32).reshape(0, 2)

        # 最终的终止接触检查列表
        self.termination_check = self.termination_contact

    def apply_action(self, actions, state):
        """应用动作到环境。

        处理输入的动作：
        1. 限制动作范围（避免极端值导致数值不稳定）
        2. 记录当前速度和动作（用于后续计算加速度和动作变化率）
        3. 根据PD控制器计算关节扭矩
        4. 将扭矩应用到仿真中

        Args:
            actions (np.ndarray): 神经网络输出的原始动作，形状 (num_envs, 12)
                                范围: [-1, 1] (标准化范围)
            state (NpEnvState): 环境状态对象，包含data和info

        Returns:
            NpEnvState: 更新后的状态对象
        """
        # =============== 步骤1：裁剪动作范围 ===============
        # 防止极端动作导致计算不稳定
        # 若超出[-1, 1]范围则被裁剪
        actions = np.clip(actions, -1.0, 1.0)

        # =============== 步骤2：记录速度和动作历史 ===============
        # 记录当前速度（用于计算加速度）
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        # 更新动作历史：previous <- current, current <- new
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions

        # =============== 步骤3：计算并应用扭矩 ===============
        # 通过PD控制器将动作转换为关节扭矩
        state.data.actuator_ctrls = self._compute_torques(actions, state.data)

        return state

    def _compute_torques(self, actions, data):
        """使用PD控制器从动作计算关节扭矩。

        PD控制器公式：
        τ = Kp * (θ_target - θ_actual) - Kd * ω_actual

        其中：
        - θ_target = action * action_scale + default_angle
        - θ_actual = 当前关节角度
        - ω_actual = 当前关节速度
        - Kp = 刚度系数
        - Kd = 阻尼系数

        Args:
            actions (np.ndarray): 标准化动作，形状 (num_envs, 12)，范围 [-1, 1]
            data (mtx.SceneData): 仿真数据对象，包含当前状态信息

        Returns:
            np.ndarray: 计算得到的关节扭矩，形状 (num_envs, 12)，范围 [-80, 80] N*m
        """
        # =============== 步骤1：缩放动作到目标位置 ===============
        # action_scale = 0.25：[-1, 1] → [-0.25, 0.25]弧度
        actions_scaled = actions * self.cfg.control_cfg.action_scale

        # =============== 步骤2：计算关节扭矩 ===============
        # P项：位置误差 * 刚度
        # 目标位置 = 缩放动作 + 默认角度
        # 位置误差 = 目标位置 - 当前位置
        position_error = actions_scaled + self.default_angles - self.get_dof_pos(data)
        p_term = self.kps * position_error

        # D项：速度阻尼
        # 抵消当前关节速度，加速度衰减
        d_term = self.kds * self.get_dof_vel(data)

        # 合成扭矩
        torques = p_term - d_term

        # =============== 步骤3：限制扭矩幅度 ===============
        # 防止扭矩过大导致仿真不稳定
        # 限制范围：[-80, 80] N*m
        torques = np.clip(torques, -80.0, 80.0)

        return torques

    def get_local_linvel(self, data: mtx.SceneData) -> np.ndarray:
        """获取本地坐标系下的线速度（速度传感器）。

        从IMU传感器或仿真中获取机器人相对于自身坐标系的线速度。
        本地坐标系中：x=前方，y=左方，z=上方。

        该速度已经经过机器人朝向的旋转，使得当机器人转向时，
        速度向量也相应旋转，始终相对于机器人的前向。

        Args:
            data (mtx.SceneData): 仿真数据对象

        Returns:
            np.ndarray: 线速度，形状 (num_envs, 3)，单位 [m/s]
                       [v_x, v_y, v_z] 在机器人坐标系中
        """
        return self._model.get_sensor_value(self.cfg.sensor_cfg.local_linvel, data)

    def get_gyro(self, data: mtx.SceneData) -> np.ndarray:
        """获取陀螺仪数据（角速度）。

        从IMU陀螺仪传感器获取机器人相对于自身坐标系的角速度。
        本地坐标系中：x=横滚(roll)，y=俯仰(pitch)，z=偏航(yaw)。

        陀螺仪数据表示机器人围绕各坐标轴的旋转速度，
        用于观测机器人的旋转运动和平衡状态。

        Args:
            data (mtx.SceneData): 仿真数据对象

        Returns:
            np.ndarray: 角速度，形状 (num_envs, 3)，单位 [rad/s]
                       [ω_x, ω_y, ω_z] 在机器人坐标系中
                       - ω_x: 横滚角速度（绕前向轴旋转）
                       - ω_y: 俯仰角速度（绕左向轴旋转）
                       - ω_z: 偏航角速度（绕上向轴旋转）
        """
        return self._model.get_sensor_value(self.cfg.sensor_cfg.gyro, data)

    def update_state(self, state):
        """更新环境状态（综合更新）。

        该方法在每个仿真步长后调用，完整地更新环境状态。
        执行顺序很重要：
        1. 先更新观测（用于计算奖励和终止条件）
        2. 再检查终止条件
        3. 最后计算奖励

        Args:
            state (NpEnvState): 环境状态对象

        Returns:
            NpEnvState: 完整更新后的状态对象
                       （包含观测、终止标志和奖励）
        """
        # 步骤1：更新观测向量和脚部接触信息
        state = self.update_observation(state)

        # 步骤2：检查回合是否应该终止
        state = self.update_terminated(state)

        # 步骤3：计算奖励
        state = self.update_reward(state)

        return state

    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """计算观测向量。

        观测向量是一个48维的向量，通过以下步骤构建：
        1. 获取传感器数据（速度、角速度）
        2. 获取机器人姿态（位置和朝向）
        3. 计算局部坐标系下的观测量（重力投影、关节偏差）
        4. 进行归一化处理
        5. 计算导航命令（相对目标位置）
        6. 组合所有观测分量

        Args:
            data (mtx.SceneData): 仿真当前帧的数据
            info (dict): 环境信息字典，包含目标位置/朝向等

        Returns:
            np.ndarray: 观测向量，形状 (num_envs, 48)

        观测向量结构：
        [0:3]     - 线速度 (归一化)
        [3:6]     - 陀螺仪 (归一化)
        [6:9]     - 重力向量投影 (未归一化)
        [9:21]    - 关节角度偏差 (归一化)
        [21:33]   - 关节速度 (归一化)
        [33:45]   - 上一步动作
        [45:48]   - 导航命令
        """
        # =============== 获取原始传感器数据 ===============
        # 基地线速度（局部坐标系）[m/s]
        linear_vel = self.get_local_linvel(data)  # (num_envs, 3)
        # 角速度/陀螺仪（局部坐标系）[rad/s]
        gyro = self.get_gyro(data)  # (num_envs, 3)

        # =============== 获取机器人姿态 ===============
        # 位置和朝向：pose = [x, y, z, qx, qy, qz, qw]
        pose = self._body.get_pose(data)  # (num_envs, 7)
        base_quat = pose[:, 3:7]  # 四元数部分

        # =============== 计算局部坐标系观测 ===============
        # 将世界坐标系的重力向量转换到机器人局部坐标系
        # 这给出了机器人相对于重力的倾斜方向
        local_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)  # (num_envs, 3)

        # 计算关节角度与默认角度的偏差
        dof_pos = self.get_dof_pos(data)  # (num_envs, 12)
        diff = dof_pos - self.default_angles  # (num_envs, 12)

        # =============== 进行归一化处理 ===============
        # 将各观测量缩放到合理范围（约[-1, 1]），加快神经网络训练
        # lin_vel归一化系数 = 2.0
        noisy_linvel = linear_vel * self.cfg.normalization_cfg.lin_vel  # (num_envs, 3)
        # ang_vel归一化系数 = 0.25
        noisy_gyro = gyro * self.cfg.normalization_cfg.ang_vel  # (num_envs, 3)
        # dof_pos归一化系数 = 1.0（一般保持原值）
        noisy_joint_angle = diff * self.cfg.normalization_cfg.dof_pos  # (num_envs, 12)
        # dof_vel归一化系数 = 0.05
        noisy_joint_vel = self.get_dof_vel(data) * self.cfg.normalization_cfg.dof_vel  # (num_envs, 12)

        # =============== 计算导航命令 ===============
        # 计算相对于机器人的目标位置和朝向偏差
        nav_command = self._compute_nav_command(data, info)  # (num_envs, 3)
        # 对导航命令进行归一化
        command = nav_command * self.commands_scale  # (num_envs, 3)

        # =============== 上一步动作 ===============
        last_actions = info["current_actions"]  # (num_envs, 12)

        # =============== 组合观测向量 ===============
        obs = np.hstack(
            [
                noisy_linvel,        # [0:3]
                noisy_gyro,          # [3:6]
                local_gravity,       # [6:9]
                noisy_joint_angle,   # [9:21]
                noisy_joint_vel,     # [21:33]
                last_actions,        # [33:45]
                command,             # [45:48]
            ]
        )
        return obs

    def _compute_nav_command(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """计算导航命令（机器人坐标系下的相对目标位置）。

        导航命令告诉策略网络目标相对于机器人的位置和朝向。
        通过将目标从世界坐标系转换到机器人坐标系来实现。

        坐标系变换步骤：
        1. 计算世界坐标系中的相对位置向量
        2. 通过四元数将其旋转到机器人坐标系
        3. 计算相对朝向角度（目标航向 - 机器人航向）

        Args:
            data (mtx.SceneData): 仿真数据，包含机器人当前位置/朝向
            info (dict): 环境信息，包含目标位置和目标朝向

        Returns:
            np.ndarray: 导航命令向量，形状 (num_envs, 3)
                       [相对x位置, 相对y位置, 相对朝向]
                       坐标系：机器人局部坐标系（x前，y左，z上）
        """
        # =============== 获取机器人当前状态 ===============
        pose = self._body.get_pose(data)  # (num_envs, 7)
        robot_pos = pose[:, :3]  # (num_envs, 3) 位置
        robot_quat = pose[:, 3:7]  # (num_envs, 4) 四元数

        # =============== 获取目标信息 ===============
        target_pos = info["target_position"]  # (num_envs, 2) - x, y（世界坐标）
        target_heading = info["target_heading"]  # (num_envs,) 目标朝向 [rad]

        # =============== 步骤1：计算世界坐标系相对位置 ===============
        # 创建3D向量容器（z=0，只关心水平面）
        rel_pos_world = np.zeros((robot_pos.shape[0], 3), dtype=np.float32)
        rel_pos_world[:, 0] = target_pos[:, 0] - robot_pos[:, 0]  # Δx
        rel_pos_world[:, 1] = target_pos[:, 1] - robot_pos[:, 1]  # Δy
        # z分量保持为0（平面导航）

        # =============== 步骤2：旋转到机器人坐标系 ===============
        # 使用四元数的反向旋转将世界坐标转换到机器人坐标
        # 这样相对位置就是以机器人为中心、机器人朝向为参考的坐标
        rel_pos_robot = np.zeros_like(rel_pos_world)
        for i in range(robot_pos.shape[0]):
            # quat_rotate_inverse将向量从世界坐标转到机器人坐标
            rel_pos_robot[i] = quat_rotate_inverse(robot_quat[i:i+1], rel_pos_world[i])

        # =============== 步骤3：计算相对朝向 ===============
        # 从四元数提取机器人的yaw角（绕z轴的旋转）
        robot_heading = self._get_heading(robot_quat)  # (num_envs,)
        # 相对朝向 = 目标朝向 - 机器人朝向
        rel_heading = target_heading - robot_heading
        # 将角度规范化到 [-π, π]
        # arctan2(sin(θ), cos(θ)) 自动处理角度的周期性
        rel_heading = np.arctan2(np.sin(rel_heading), np.cos(rel_heading))

        # =============== 步骤4：组合导航命令 ===============
        nav_command = np.zeros((robot_pos.shape[0], 3), dtype=np.float32)
        nav_command[:, 0] = rel_pos_robot[:, 0]  # 机器人坐标系中的x偏差 (前方)
        nav_command[:, 1] = rel_pos_robot[:, 1]  # 机器人坐标系中的y偏差 (左方)
        nav_command[:, 2] = rel_heading           # 相对朝向 [-π, π]

        return nav_command

    def _get_heading(self, quat: np.ndarray) -> np.ndarray:
        """从四元数提取yaw角（绕z轴旋转）。

        Yaw角代表机器人在水平面上的朝向，范围 [-π, π]。
        该方法从四元数中提取yaw分量，忽略roll和pitch。

        四元数到欧拉角的标准转换公式（仅提取yaw）：
        yaw = arctan2(2*(w*z + x*y), 1 - 2*(y² + z²))

        Args:
            quat (np.ndarray): 四元数数组，形状 (num_envs, 4)，格式 [x, y, z, w]

        Returns:
            np.ndarray: Yaw角数组，形状 (num_envs,)，范围 [-π, π]

        参考资料：
        四元数 q = [x, y, z, w] 表示旋转矩阵R的转换
        对于标准旋转矩阵：
        R = [[cos(yaw), -sin(yaw), 0],
             [sin(yaw),  cos(yaw), 0],
             [0,         0,        1]]
        其中yaw = arctan2(R[1,0], R[0,0])
        """
        # =============== 提取四元数分量 ===============
        # 四元数格式：[x, y, z, w]（虚部+实部）
        x = quat[:, 0]
        y = quat[:, 1]
        z = quat[:, 2]
        w = quat[:, 3]

        # =============== 四元数到yaw角的转换 ===============
        # 分子：2 * (w*z + x*y)
        # 这对应旋转矩阵R[1,0]（经过归一化）
        siny_cosp = 2 * (w * z + x * y)

        # 分母：1 - 2*(y² + z²)
        # 这对应旋转矩阵R[0,0]（经过归一化）
        cosy_cosp = 1 - 2 * (y * y + z * z)

        # =============== 计算yaw角 ===============
        # arctan2(y, x) 返回 [-π, π] 范围内的角度
        # 这是标准的反正切函数，能正确处理所有象限
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return yaw

    def update_observation(self, state: NpEnvState):
        """更新状态中的观测向量和脚部接触信息。

        该方法执行以下任务：
        1. 计算当前帧的观测向量（48维）
        2. 检测脚与地面的接触
        3. 更新脚部悬空时间追踪

        Args:
            state (NpEnvState): 环境状态对象

        Returns:
            NpEnvState: 更新后的状态对象（包含新观测）
        """
        data = state.data

        # =============== 步骤1：计算观测向量 ===============
        obs = self._get_obs(data, state.info)

        # =============== 步骤2：检测脚部接触 ===============
        if self.foot_check_num > 0:
            # 获取当前帧的接触查询对象
            cquerys = self._model.get_contact_query(data)
            # 检查脚与地面的接触关系
            # is_colliding返回接触矩阵，形状 (total_contacts,)
            foot_contact = cquerys.is_colliding(self.foot_check)
            # 重塑为 (num_envs, foot_check_num)
            state.info["contacts"] = foot_contact.reshape((self._num_envs, self.foot_check_num))
            # 更新脚部悬空时间
            state.info["feet_air_time"] = self.update_feet_air_time(state.info)
        else:
            # 若无脚部定义，初始化为全False
            state.info["contacts"] = np.zeros((self._num_envs, 4), dtype=np.bool_)
            state.info["feet_air_time"] = np.zeros((self._num_envs, 4), dtype=np.float32)

        # 返回更新观测后的状态
        return state.replace(obs=obs)

    def update_terminated(self, state: NpEnvState) -> NpEnvState:
        """检查回合终止条件。

        环境在以下情况下终止：
        1. 身体与地面接触（即机器人摔倒）
        2. 身体高度过低（下沉）
        3. 机器人倾斜过大（翻滚）
        4. 关节速度过高（仿真不稳定）

        回合终止是强化学习的一个重要信号，用于：
        - 防止不现实的状态
        - 加速训练收敛
        - 避免仿真不稳定

        Args:
            state (NpEnvState): 环境状态对象

        Returns:
            NpEnvState: 更新后的状态对象（包含terminated标志）
        """
        data = state.data

        # =============== 条件1：身体接触检测 ===============
        # 检查身体是否与地面接触（配置中的 terminate_after_contacts_on）
        if self.num_check > 0:
            # 获取接触查询对象
            cquerys = self._model.get_contact_query(data)
            # 检查所有回合终止接触对是否发生接触
            termination_check = cquerys.is_colliding(self.termination_check)
            # 重塑为 (num_envs, num_check)
            termination_check = termination_check.reshape((self._num_envs, self.num_check))
            # 如果任何接触对接触，则该环境终止
            contact_terminated = termination_check.any(axis=1)  # (num_envs,)
        else:
            contact_terminated = np.zeros(self._num_envs, dtype=np.bool_)

        # =============== 条件2：高度检查 ===============
        # 机器人身体高度过低表示已摔倒
        pose = self._body.get_pose(data)  # (num_envs, 7)
        base_height = pose[:, 2]  # (num_envs,) z坐标
        # 阈值0.3m：当站立姿态高度约0.62m时，0.3m表示显著下沉
        height_terminated = base_height < 0.3

        # =============== 条件3：姿态检查 ===============
        # 机器人倾斜过大表示可能翻滚或失控
        base_quat = pose[:, 3:7]  # (num_envs, 4)
        # 将世界重力向量转换到机器人坐标系
        local_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)  # (num_envs, 3)
        # 检查重力向量的z分量
        # 正常站立时：local_gravity ≈ [0, 0, -1]
        # 倾斜过大时：local_gravity[:, 2] > -0.5（即z分量上升）
        # 这意味着重力向量指向过于水平，即机器人倾斜超过约60°
        orientation_terminated = local_gravity[:, 2] > -0.5

        # =============== 条件4：关节速度检查 ===============
        # 关节速度过高表示仿真可能不稳定或发生异常
        joint_vel = self.get_dof_vel(data)  # (num_envs, 12)
        # 找到每个环境中最大的关节速度（取绝对值）
        max_joint_vel = np.max(np.abs(joint_vel), axis=1)  # (num_envs,)
        # 阈值50.0 rad/s：超过此值表示关节运动过快，可能导致数值不稳定
        velocity_terminated = max_joint_vel > 50.0

        # =============== 合并所有终止条件 ===============
        # 任何条件为真则回合终止
        terminated = contact_terminated | height_terminated | orientation_terminated | velocity_terminated

        return state.replace(terminated=terminated)

    def update_feet_air_time(self, info: dict):
        """更新脚部悬空时间追踪。

        脚部悬空时间用于奖励函数，鼓励机器人拥有适当的步频。
        当脚接触地面时，悬空时间重置为0；
        当脚离开地面时，悬空时间逐帧递增。

        算法：
        1. 对所有脚部的悬空时间加上一个时间步长 (ctrl_dt)
        2. 如果脚已接触地面，重置该脚的悬空时间为0

        Args:
            info (dict): 环境信息字典，包含feet_air_time和contacts

        Returns:
            np.ndarray: 更新后的脚部悬空时间，形状 (num_envs, num_feet)
        """
        feet_air_time = info["feet_air_time"]  # (num_envs, num_feet)
        # 步骤1：所有脚的悬空时间增加一个控制周期
        feet_air_time += self.cfg.ctrl_dt  # ctrl_dt = 0.02s

        # 步骤2：如果脚接触地面，重置为0
        # ~info["contacts"]：取反，True→False (接触), False→True (未接触)
        # 所以这行的效果是：如果接触则乘以0（重置），否则保持原值
        feet_air_time *= ~info["contacts"]

        return feet_air_time

    def resample_commands(self, num_envs: int):
        """重新采样速度命令（用于运动控制）。

        在每个回合开始时或定期地重新采样速度命令。
        这告诉机器人应该如何运动（目标速度）。

        采样范围：
        - 前进速度 vel_x: [0, 2.0] m/s
        - 横向速度 vel_y: [-1.0, 1.0] m/s
        - 角速度 ang_vel: [-1.0, 1.0] rad/s

        Args:
            num_envs (int): 环境数量

        Returns:
            np.ndarray: 速度命令，形状 (num_envs, 3)，单位 [m/s, m/s, rad/s]
        """
        commands = np.random.uniform(
            low=self.cfg.commands_cfg.vel_limit[0],   # [0, -1.0, -1.0]
            high=self.cfg.commands_cfg.vel_limit[1],  # [2.0, 1.0, 1.0]
            size=(num_envs, 3),
        )
        return commands

    def resample_target(self, num_envs: int):
        """重新采样导航目标位置和朝向。

        在每个回合开始时为每个环境生成一个新的目标位置和朝向。
        这定义了机器人在该回合内应该到达的地点。

        采样范围：
        - 目标位置: x ∈ [-5, 5]m, y ∈ [-5, 5]m（10m × 10m的正方形区域）
        - 目标朝向: [-π, π]（任意方向）

        Args:
            num_envs (int): 环境数量

        Returns:
            tuple: (target_pos, target_heading)
                - target_pos: 目标位置，形状 (num_envs, 2)，单位 [m, m]
                - target_heading: 目标朝向，形状 (num_envs,)，单位 [rad]
        """
        pos_range = self.cfg.commands_cfg.pos_range  # [[-5, -5], [5, 5]]

        # 采样目标位置：在指定范围内均匀采样
        target_pos = np.random.uniform(
            low=pos_range[0],    # [-5.0, -5.0]
            high=pos_range[1],   # [5.0, 5.0]
            size=(num_envs, 2),
        ).astype(np.float32)

        # 采样目标朝向：在 [-π, π] 范围内均匀采样
        target_heading = np.random.uniform(
            low=-np.pi,
            high=np.pi,
            size=(num_envs,),
        ).astype(np.float32)

        return target_pos, target_heading

    def update_reward(self, state: NpEnvState) -> NpEnvState:
        """计算环境奖励。

        奖励计算流程：
        1. 计算各个奖励分量（导航奖励、运动奖励等）
        2. 根据配置的权重进行加权求和
        3. 对回合终止的环境施加终止惩罚
        4. 限制奖励范围以避免数值爆炸

        奖励=导航奖励+运动奖励+终止惩罚

        Args:
            state (NpEnvState): 环境状态对象

        Returns:
            NpEnvState: 更新后的状态对象（包含reward字段）
        """
        data = state.data
        terminated = state.terminated  # (num_envs,)

        # =============== 步骤1：计算所有奖励分量 ===============
        # _get_reward返回一个字典，包含所有奖励项的原始值（未加权）
        reward_dict = self._get_reward(data, state.info)

        # =============== 步骤2：根据权重加权奖励 ===============
        # 获取配置中该奖励项的权重系数，若无定义则为0
        rewards = {
            k: v * self.cfg.rewards_cfg.scales.get(k, 0.0)
            for k, v in reward_dict.items()
        }

        # =============== 步骤3：求和得到总奖励 ===============
        rwd = sum(rewards.values())  # (num_envs,)

        # =============== 步骤4：限制奖励范围 ===============
        # 防止由于数值问题导致奖励爆炸
        # 范围: [0.0, 10000.0]
        rwd = np.clip(rwd, 0.0, 10000.0)

        # =============== 步骤5：添加终止惩罚 ===============
        # 对于已终止的环境，添加额外的终止惩罚（通常为负值）
        if "termination_penalty" in self.cfg.rewards_cfg.scales:
            rwd = apply_termination_penalty(
                rwd,
                terminated,
                self.cfg.rewards_cfg.scales["termination_penalty"],
            )
        else:
            rwd = np.where(terminated, np.array(0.0), rwd)

        return state.replace(reward=rwd)

    def reset(self, data) -> tuple[np.ndarray, dict]:
        """重置环境到初始状态。

        重置过程包括：
        1. 设置机器人的初始位置、朝向和关节角度
        2. 清除所有速度（静止开始）
        3. 重新采样导航目标和速度命令
        4. 初始化所有追踪信息（动作历史、接触状态等）
        5. 计算初始观测

        Args:
            data (mtx.SceneData): 仿真数据对象

        Returns:
            tuple: (obs, info)
                - obs: 初始观测向量，形状 (num_envs, 48)
                - info: 环境信息字典，包含命令、目标位置、追踪信息等
        """
        num_reset = data.shape[0]  # 获取重置环境的数量

        # =============== 步骤1：准备初始关节位置和速度 ===============
        # 复制初始位置到所有环境实例
        # _init_dof_pos = [base_x, base_y, base_z, qx, qy, qz, qw, joint_angles(12)]
        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1))  # (num_reset, 19)

        # 初始速度全为0（静止开始）
        # _init_dof_vel = [base_lin_vel(3), base_ang_vel(3), joint_vel(12)]
        dof_vel = np.tile(self._init_dof_vel, (num_reset, 1))  # (num_reset, 18)

        # =============== 步骤2：重置仿真状态 ===============
        # 重置仿真数据到基础状态
        data.reset(self._model)

        # 设置关节速度
        data.set_dof_vel(dof_vel)

        # 设置关节位置（包括base位置和朝向）
        data.set_dof_pos(dof_pos, self._model)

        # 进行正向运动学计算（更新all_pos, all_vel等）
        self._model.forward_kinematic(data)

        # =============== 步骤3：重新采样导航目标和命令 ===============
        # 为每个环境采样新的导航目标
        target_pos, target_heading = self.resample_target(num_reset)

        # =============== 步骤4：初始化信息字典 ===============
        info = {
            # ============ 动作追踪 ============
            # 当前执行的动作（初始为0）
            "current_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            # 上一步执行的动作（初始为0）
            "last_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),

            # ============ 运动命令 ============
            # 速度命令：机器人应该如何运动
            "commands": self.resample_commands(num_reset),

            # ============ 导航目标 ============
            # 目标位置（2D，在世界坐标系）
            "target_position": target_pos,
            # 目标朝向（机器人应该面向的方向）
            "target_heading": target_heading,

            # ============ 关节速度追踪 ============
            # 上一步的关节速度（用于计算加速度）
            "last_dof_vel": np.zeros((num_reset, self._num_action), dtype=np.float32),

            # ============ 脚部状态 ============
            # 脚部悬空时间：每条腿已离开地面的时间
            # 如果没有脚部定义，使用4作为默认值
            "feet_air_time": np.zeros((num_reset, max(self.foot_check_num, 4)), dtype=np.float32),
            # 脚部接触状态：是否与地面接触
            "contacts": np.zeros((num_reset, max(self.foot_check_num, 4)), dtype=np.bool_),
        }

        # =============== 步骤5：计算初始观测 ===============
        obs = self._get_obs(data, info)

        return obs, info

    def _get_reward(
        self,
        data: mtx.SceneData,
        info: dict,
    ) -> dict[str, np.ndarray]:
        """计算所有奖励项。

        返回一个字典，包含所有未加权的奖励项。
        这些奖励项稍后会根据配置中的权重进行加权求和。

        奖励分为两大类：
        1. 导航奖励：鼓励机器人到达目标位置和朝向
        2. 运动奖励：鼓励高效、平滑的运动

        Args:
            data (mtx.SceneData): 仿真数据，包含当前物理状态
            info (dict): 环境信息，包含命令、目标等

        Returns:
            dict[str, np.ndarray]: 奖励项字典，每个值的形状为 (num_envs,)

        奖励项说明：
        ┌─── 导航奖励 ────────────────────────────────────┐
        │ position_tracking: 粗粒度位置追踪（容差2.0m）    │
        │ position_tracking_fine_grained: 精细位置追踪（容差0.2m）│
        │ orientation_tracking: 朝向追踪                      │
        ├─── 运动奖励 ────────────────────────────────────┤
        │ tracking_lin_vel: 线速度命令跟踪               │
        │ tracking_ang_vel: 角速度命令跟踪               │
        │ feet_air_time: 脚部悬空时间（步态质量）       │
        │ lin_vel_z: 竖直速度惩罚（不应跳跃）           │
        │ ang_vel_xy: 横滚俯仰速度惩罚（保持平衡）      │
        │ orientation: 姿态惩罚（保持竖直）             │
        │ torques: 扭矩惩罚（能量效率）                  │
        │ dof_vel: 关节速度惩罚（平滑运动）             │
        │ dof_acc: 关节加速度惩罚（更平滑）             │
        │ action_rate: 动作变化率惩罚（连续性）         │
        │ stand_still: 静止惩罚（低速时应静止）        │
        │ hip_pos: 髋关节位置惩罚（避免极端配置）      │
        └──────────────────────────────────────────────────┘
        """
        commands = info["commands"]  # (num_envs, 3)

        return {
            # ==================== 导航相关奖励 ====================
            # 粗粒度位置追踪：鼓励到达目标区域（容差较大）
            "position_tracking": self._reward_position_tracking(data, info),
            # 精细位置追踪：精确到达目标点（容差小）
            "position_tracking_fine_grained": self._reward_position_tracking_fine_grained(data, info),
            # 朝向追踪：奖励与目标朝向对齐
            "orientation_tracking": self._reward_orientation_tracking(data, info),

            # ==================== 运动控制相关奖励 ====================
            # 竖直线速度惩罚：禁止跳跃
            "lin_vel_z": self._reward_lin_vel_z(data),
            # 横滚俯仰角速度惩罚：保持平衡
            "ang_vel_xy": self._reward_ang_vel_xy(data),
            # 姿态惩罚：保持竖直站立
            "orientation": self._reward_orientation(data),
            # 扭矩惩罚：鼓励高效运动（低能耗）
            "torques": self._reward_torques(data),
            # 关节速度惩罚：鼓励缓慢、平滑运动
            "dof_vel": self._reward_dof_vel(data),
            # 关节加速度惩罚：鼓励更平滑的加速
            "dof_acc": self._reward_dof_acc(data, info),
            # 动作变化率惩罚：鼓励连续的控制输出
            "action_rate": self._reward_action_rate(info),
            # 线速度追踪：跟踪给定的前进速度和横向速度
            "tracking_lin_vel": self._reward_tracking_lin_vel(data, commands),
            # 角速度追踪：跟踪给定的旋转速度
            "tracking_ang_vel": self._reward_tracking_ang_vel(data, commands),
            # 静止奖励：在低速命令下保持静止
            "stand_still": self._reward_stand_still(data, commands),
            # 髋关节位置奖励：避免腿部过度伸展
            "hip_pos": self._reward_hip_pos(data, commands),
            # 脚部悬空时间奖励：鼓励好的步态
            "feet_air_time": self._reward_feet_air_time(commands, info),
        }

    # =============== 导航奖励函数 ===============

    def _reward_position_tracking(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """粗粒度位置追踪奖励（tanh核函数）。

        使用tanh核函数奖励靠近目标位置的行为。
        tanh函数的特点：
        - 在距离小时梯度大（奖励快速）
        - 在距离大时梯度小（避免饱和）
        - 范围：[0, 1]

        奖励公式：
        r = 1 - tanh(distance / std)

        其中std=2.0m（粗粒度容差），表示：
        - distance=0时，r=1.0（完全奖励）
        - distance=2.0时，r≈0.76（部分奖励）
        - distance→∞时，r→0（无奖励）

        Args:
            data (mtx.SceneData): 仿真数据
            info (dict): 包含target_position的信息字典

        Returns:
            np.ndarray: 粗粒度位置追踪奖励，形状 (num_envs,)
        """
        pose = self._body.get_pose(data)
        robot_pos = pose[:, :2]  # (num_envs, 2) x, y坐标
        target_pos = info["target_position"]  # (num_envs, 2)

        # 计算欧氏距离
        distance = np.linalg.norm(target_pos - robot_pos, axis=1)  # (num_envs,)

        # 从配置中获取标准差（容差）
        std = self.cfg.rewards_cfg.position_tracking_std  # 2.0 m

        # tanh奖励函数
        return 1 - np.tanh(distance / std)

    def _reward_position_tracking_fine_grained(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """精细位置追踪奖励。

        与粗粒度位置追踪类似，但使用更小的容差（std=0.2m）。
        这意味着该奖励只在机器人非常靠近目标时才有显著奖励。

        奖励公式：
        r = 1 - tanh(distance / std)

        其中std=0.2m（精细容差），表示：
        - distance=0时，r=1.0
        - distance=0.2时，r≈0.76
        - distance>1.0时，r≈0（基本无奖励）

        权重通常设置为1.5，比粗粒度追踪权重(1.0)更高，
        鼓励机器人精确到达目标。

        Args:
            data (mtx.SceneData): 仿真数据
            info (dict): 包含target_position的信息字典

        Returns:
            np.ndarray: 精细位置追踪奖励，形状 (num_envs,)
        """
        pose = self._body.get_pose(data)
        robot_pos = pose[:, :2]
        target_pos = info["target_position"]
        distance = np.linalg.norm(target_pos - robot_pos, axis=1)

        # 使用更小的标准差
        std = self.cfg.rewards_cfg.position_tracking_fine_grained_std  # 0.2 m
        return 1 - np.tanh(distance / std)

    def _reward_orientation_tracking(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """朝向追踪奖励（实际为惩罚）。

        奖励机器人面向正确的方向（目标朝向）。
        通过计算朝向误差的绝对值，误差越小奖励越高。

        奖励公式（实际为惩罚）：
        r = |heading_error|  (范围 [0, π])

        该项通常与负权重相乘，形成惩罚：
        权重 = -0.1，所以实际奖励 = -0.1 * |heading_error|

        这是一个较软的约束，只轻微惩罚朝向错误。

        Args:
            data (mtx.SceneData): 仿真数据
            info (dict): 包含target_heading的信息字典

        Returns:
            np.ndarray: 朝向误差，形状 (num_envs,)，范围 [0, π]
        """
        pose = self._body.get_pose(data)
        robot_quat = pose[:, 3:7]

        # 从四元数提取机器人的yaw角
        robot_heading = self._get_heading(robot_quat)  # (num_envs,)
        target_heading = info["target_heading"]  # (num_envs,)

        # 计算朝向误差
        heading_error = target_heading - robot_heading  # (num_envs,)

        # 将角度规范化到 [-π, π]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # 返回绝对误差
        return np.abs(heading_error)

    # =============== 运动控制奖励函数 ===============

    def _reward_lin_vel_z(self, data):
        """惩罚竖直线速度（禁止跳跃和下沉）。

        机器人应该在水平面上移动，不应该在竖直方向上有显著速度。
        跳跃和下沉都会被惩罚。

        惩罚公式：
        r = v_z²

        这是一个平方惩罚，小的竖直速度影响小，大的竖直速度影响大。
        权重通常为-2.0，强烈抑制竖直运动。

        Args:
            data (mtx.SceneData): 仿真数据

        Returns:
            np.ndarray: 竖直速度平方，形状 (num_envs,)
        """
        # 获取本地坐标系下的线速度（最后一维是z）
        return np.square(self.get_local_linvel(data)[:, 2])

    def _reward_ang_vel_xy(self, data):
        """惩罚横滚和俯仰角速度（保持平衡）。

        机器人应该保持稳定，不应该有过大的横滚(roll)或俯仰(pitch)角速度。
        这鼓励机器人保持平衡，避免翻滚。

        惩罚公式：
        r = w_x² + w_y²

        权重通常为-0.05，是一个较软的约束。

        Args:
            data (mtx.SceneData): 仿真数据

        Returns:
            np.ndarray: 横滚俯仰角速度平方和，形状 (num_envs,)
        """
        # 获取陀螺仪数据，取前两个分量（x, y）
        gyro = self.get_gyro(data)  # (num_envs, 3)
        return np.sum(np.square(gyro[:, :2]), axis=1)

    def _reward_orientation(self, data):
        """惩罚非水平的基座朝向（鼓励保持竖直姿态）。

        机器人的身体应该相对竖直，不应该向前倾或向后倾。
        通过检查重力向量在机器人坐标系中的水平分量来判断。

        惩罚公式：
        r = g_x² + g_y²

        其中[g_x, g_y, g_z]是重力向量在机器人坐标系中的投影。
        当机器人竖直时，只有g_z=-1，g_x和g_y接近0。
        权重为-0.0（实际不起作用），如需启用则改为负值。

        Args:
            data (mtx.SceneData): 仿真数据

        Returns:
            np.ndarray: 姿态误差，形状 (num_envs,)
        """
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        # 获取重力在机器人坐标系中的投影
        gravity = quat_rotate_inverse(base_quat, self.gravity_vec)  # (num_envs, 3)
        # 计算x和y分量的平方和（代表偏离竖直的程度）
        return np.sum(np.square(gravity[:, :2]), axis=1)

    def _reward_torques(self, data: mtx.SceneData):
        """惩罚关节扭矩（鼓励能量效率）。

        过大的扭矩表示需要用力，消耗能量。
        这个惩罚鼓励机器人使用低扭矩完成动作。

        惩罚公式：
        r = Σ(τ²)

        权重通常为-0.00001（非常小），提供一个轻微的能耗考虑。

        Args:
            data (mtx.SceneData): 仿真数据，包含actuator_ctrls

        Returns:
            np.ndarray: 扭矩平方和，形状 (num_envs,)
        """
        # actuator_ctrls是实际施加的关节扭矩
        return np.sum(np.square(data.actuator_ctrls), axis=1)

    def _reward_dof_vel(self, data):
        """惩罚关节速度（鼓励缓慢、平滑运动）。

        过高的关节速度会导致快速、剧烈的运动，这通常不是我们想要的。
        权重为-0.0（实际不起作用），如需启用则改为负值。

        惩罚公式：
        r = Σ(ω²)

        Args:
            data (mtx.SceneData): 仿真数据

        Returns:
            np.ndarray: 关节速度平方和，形状 (num_envs,)
        """
        return np.sum(np.square(self.get_dof_vel(data)), axis=1)

    def _reward_dof_acc(self, data, info):
        """惩罚关节加速度（鼓励更平滑的加速）。

        急剧的加速度变化会导致运动不平滑，可能导致步态不稳定。
        这个惩罚鼓励关节加速度的变化更平缓。

        惩罚公式：
        r = Σ((ω_current - ω_last)²) / Δt²

        权重通常为-2.5e-7（极小），只是提供轻微的平滑约束。

        Args:
            data (mtx.SceneData): 仿真数据
            info (dict): 包含上一步速度的信息字典

        Returns:
            np.ndarray: 加速度平方和，形状 (num_envs,)
        """
        # 计算加速度 = (当前速度 - 上一速度) / Δt
        dof_acc = (info["last_dof_vel"] - self.get_dof_vel(data)) / self.cfg.ctrl_dt
        return np.sum(np.square(dof_acc), axis=1)

    def _reward_action_rate(self, info: dict):
        """惩罚动作变化率（鼓励连续的控制输出）。

        相邻两步之间的动作变化过大会导致控制输出不连续。
        这个惩罚鼓励策略输出平滑的、连续的动作。

        惩罚公式：
        r = Σ((a_current - a_last)²)

        权重通常为-0.001，在众多奖励中影响中等。

        Args:
            info (dict): 包含当前和上一步动作的信息字典

        Returns:
            np.ndarray: 动作变化平方和，形状 (num_envs,)
        """
        action_diff = info["current_actions"] - info["last_actions"]  # (num_envs, 12)
        return np.sum(np.square(action_diff), axis=1)

    def _reward_termination(self, done):
        """回合终止的奖励/惩罚。

        当回合终止时返回1.0，否则返回0.0。
        这个值会乘以negative权重（-200.0），形成重大惩罚。

        Args:
            done (np.ndarray): 布尔数组，表示每个环境是否已终止

        Returns:
            np.ndarray: 终止标志转换为浮点，形状 (num_envs,)
        """
        return done.astype(np.float32)

    def _reward_feet_air_time(self, commands: np.ndarray, info: dict):
        """奖励脚部悬空时间（鼓励良好的步态）。

        机器人应该有规律的步态，脚部应该有适当的悬空时间（离地时间）。
        这鼓励机器人做出有效的行走运动而不是滑行。

        奖励逻辑：
        1. 只有当机器人在运动时（线速度>0.1m/s）才给予奖励
        2. 奖励脚部悬空时间超过0.5秒的时间（(air_time - 0.5)）
        3. 通过接触检测来确定脚何时着地

        Args:
            commands (np.ndarray): 速度命令，用于判断是否在运动
            info (dict): 包含feet_air_time和contacts的信息字典

        Returns:
            np.ndarray: 脚部悬空时间奖励，形状 (num_envs,)
        """
        feet_air_time = info["feet_air_time"]  # (num_envs, num_feet)

        if feet_air_time.shape[1] == 0:
            # 如果没有脚部定义，返回零
            return np.zeros(commands.shape[0], dtype=np.float32)

        # first_contact = True当脚既有悬空时间且当前接触地面
        # 这标记了脚刚接触地面的时刻
        first_contact = (feet_air_time > 0.0) * info["contacts"]

        # 计算奖励：奖励悬空时间超过0.5秒的部分
        rew_airTime = np.sum((feet_air_time - 0.5) * first_contact, axis=1)

        # 只有当机器人在运动时（前进速度>0.1m/s）才给予奖励
        # 这避免了静止时的误奖励
        moving = np.linalg.norm(commands[:, :2], axis=1) > 0.1
        rew_airTime *= moving

        return rew_airTime

    def _reward_tracking_lin_vel(self, data, commands: np.ndarray):
        """奖励线速度命令跟踪（奖励准确性）。

        机器人应该准确地跟踪给定的线速度命令（前进和横向）。
        使用高斯核函数：当误差小时奖励大，当误差大时奖励小。

        奖励公式：
        r = exp(-error / sigma)

        其中error = |v_target_x - v_actual_x|² + |v_target_y - v_actual_y|²
        sigma = 0.25（追踪精度参数）

        Args:
            data (mtx.SceneData): 仿真数据
            commands (np.ndarray): 目标线速度命令

        Returns:
            np.ndarray: 线速度追踪奖励，形状 (num_envs,)，范围 [0, 1]
        """
        # 计算前进和横向速度的误差
        lin_vel_error = np.sum(
            np.square(commands[:, :2] - self.get_local_linvel(data)[:, :2]),
            axis=1
        )
        # 高斯奖励函数
        return np.exp(-lin_vel_error / self.cfg.rewards_cfg.tracking_sigma)

    def _reward_tracking_ang_vel(self, data, commands: np.ndarray):
        """奖励角速度命令跟踪（yaw旋转）。

        机器人应该准确地跟踪给定的角速度命令（绕z轴旋转）。
        使用高斯核函数。

        奖励公式：
        r = exp(-error / sigma)

        其中error = (ω_target_z - ω_actual_z)²
        sigma = 0.25（追踪精度参数）

        Args:
            data (mtx.SceneData): 仿真数据
            commands (np.ndarray): 目标角速度命令

        Returns:
            np.ndarray: 角速度追踪奖励，形状 (num_envs,)，范围 [0, 1]
        """
        # 计算yaw角速度误差
        ang_vel_error = np.square(commands[:, 2] - self.get_gyro(data)[:, 2])
        # 高斯奖励函数
        return np.exp(-ang_vel_error / self.cfg.rewards_cfg.tracking_sigma)

    def _reward_stand_still(self, data, commands: np.ndarray):
        """惩罚在静止命令下的运动。

        当速度命令为零时（||commands||<0.1），机器人应该保持静止。
        如果在这种情况下仍然运动，会受到惩罚。

        惩罚公式：
        r = Σ|θ_i - θ_default_i|   （仅当 ||commands|| < 0.1时）

        这鼓励机器人在静止命令下回到默认站立姿态。

        Args:
            data (mtx.SceneData): 仿真数据
            commands (np.ndarray): 速度命令

        Returns:
            np.ndarray: 静止惩罚，形状 (num_envs,)
        """
        # 计算关节偏离默认位置的程度
        joint_deviation = np.sum(np.abs(self.get_dof_pos(data) - self.default_angles), axis=1)

        # 仅当速度命令为零时才施加惩罚
        is_standing_still = np.linalg.norm(commands, axis=1) < 0.1

        return joint_deviation * is_standing_still

    def _reward_hip_pos(self, data, commands: np.ndarray):
        """惩罚髋关节位置远离默认（避免极端配置）。

        髋关节（HAA）的过度打开或关闭会导致不稳定的步态。
        这个惩罚在横向速度大时更强（鼓励稳定的横向运动）。

        惩罚公式：
        r = (0.8 - |v_y|) * Σ(θ_hip_i - θ_default_hip_i)²

        当|v_y|小时，(0.8 - |v_y|)接近0.8（强惩罚）
        当|v_y|大时，(0.8 - |v_y|)接近0（弱惩罚，因为此时髋关节打开是必要的）

        Args:
            data (mtx.SceneData): 仿真数据
            commands (np.ndarray): 速度命令

        Returns:
            np.ndarray: 髋关节位置惩罚，形状 (num_envs,)
        """
        if len(self.hip_indices) == 0:
            return np.zeros(commands.shape[0], dtype=np.float32)

        # 基于横向速度的权重：横向运动越多，对髋关节位置的惩罚越小
        weight = 0.8 - np.abs(commands[:, 1])  # 0.8 - |v_y|

        # 计算髋关节的位置偏差
        hip_deviation = np.sum(
            np.square(
                self.get_dof_pos(data)[:, self.hip_indices] -
                self.default_angles[self.hip_indices]
            ),
            axis=1
        )

        return weight * hip_deviation
