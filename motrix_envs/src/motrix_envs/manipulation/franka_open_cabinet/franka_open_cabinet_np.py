import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import FrankaOpenCabinetEnvCfg
import copy
# 设置打印选项，保留2位小数
np.set_printoptions(precision=2)

def quaternion_rotation_reward_np(q_current, q_target):
    """
    使用 NumPy 计算两个批量四元数之间的姿态对齐奖励。

    参数:
        q_current (np.ndarray): 当前姿态的四元数，形状为 (num_envs, 4)。
        q_target (np.ndarray): 目标姿态的四元数，形状为 (num_envs, 4) 或 (4,)。
                                如果是 (4,)，则会被广播到所有环境。

    返回:
        np.ndarray: 每个环境对应的奖励值，形状为 (num_envs,)。奖励值范围在 [-1, 1] 之间。
    """
    # 确保输入是浮点型数组
    q_current = q_current.astype(np.float32)
    q_target = q_target.astype(np.float32)

    # 如果 q_target 是单个四元数，则广播到所有环境
    if q_target.ndim == 1:
        # 使用 np.tile 进行广播
        q_target = np.tile(q_target, (q_current.shape[0], 1))

    # 步骤 1: 计算 q_current 的共轭
    # 四元数 (w, x, y, z) 的共轭是 (w, -x, -y, -z)
    q_current_conj = np.copy(q_current)
    q_current_conj[..., 1:] *= -1  # 对 x, y, z 分量取反

    # 步骤 2: 计算相对四元数 q_rel = q_target * q_current_conj
    # 我们将四元数的分量解包出来以便计算
    w1, x1, y1, z1 = q_target[..., 0], q_target[..., 1], q_target[..., 2], q_target[..., 3]
    w2, x2, y2, z2 = q_current_conj[..., 0], q_current_conj[..., 1], q_current_conj[..., 2], q_current_conj[..., 3]

    # 应用四元数乘法公式
    w_rel = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_rel = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_rel = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_rel = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # 为了数值稳定性，将 w_rel clamping 到 [-1.0, 1.0] 范围内
    w_rel_clamped = np.clip(w_rel, -1.0, 1.0)

    # 步骤 3: 计算旋转角度 theta
    theta = 2.0 * np.arccos(w_rel_clamped)

    # 步骤 4: 计算奖励
    reward = np.cos(theta)

    return reward

@registry.env("franka_open_cabinet", "np")
class FrankaOpenCabinetEnv(NpEnv):
    _cfg: FrankaOpenCabinetEnvCfg

    def __init__(self, cfg: FrankaOpenCabinetEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self.robot_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'finger_joint1', 'finger_joint2']
        self.robot_default_joint_pos = np.array([0.0 * np.pi, -30/180* np.pi, 0 * np.pi, -156/180 * np.pi, 0.0 * np.pi, 186/180 * np.pi, -45/180 * np.pi, 0.04, 0.04], np.float32)
        # [ 0.  -0.5235988  0.    -2.7227137  0.     3.2463124    -0.7853982  0.04   0.04 ]

        self._action_dim = 8
        self._obs_dim = 25 # 8 + 8 + 7 + 1 + 1
        self._action_space = gym.spaces.Box(-np.inf, np.inf, (self._action_dim,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (self._obs_dim,), dtype=np.float32)

        self._num_dof_pos = 9 # self._model.num_dof_pos # 9
        self._num_dof_vel = 9 # self._model.num_dof_vel # 9
        self._init_dof_pos = self.robot_default_joint_pos
        self._init_dof_vel = np.zeros(self._num_dof_vel, dtype=np.float32)

        self._robot_actuators = [ self._model.get_actuator(name) for name in ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7', 'actuator8']]


        # 一些属性
        self.robot = self._model.get_body("link0")
        self.gripper_tcp = self._model.get_site("gripper")
        self.left_finger_pad = self._model.get_geom("left_finger_pad")
        self.right_finger_pad = self._model.get_geom("right_finger_pad")
        self.robot_joint_pos_min_limit = self._cfg.control_config.min_pos
        self.robot_joint_pos_max_limit = self._cfg.control_config.max_pos

        self.drawer_top_joint = self._model.get_joint("drawer_top_joint")
        self.drawer_top_handle = self._model.get_site("drawer_top_handle")

        # 进行统计
        self.count = 0

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        assert not np.isnan(actions).any(), "actions contain nan"

        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions

        # no gripper
        old_joint_pos = self.get_robot_joint_pos(state.data)[:, :self._action_dim-1]
        new_joint_pos = actions[:, :self._action_dim-1]  + old_joint_pos # action as offset

        # with gripper
        # 1. 映射为概率 p (使用 Sigmoid)
        probabilities = 1 / (1 + np.exp(-actions[:, -1]))
        # 2. 伯努利采样 概率总是有可能采样到不同的结果
        # np.random.uniform(0, 1, size) 为每个环境生成一个随机数 r ~ U(0, 1)
        # 如果 r < p，则结果为 1 (成功/抓取)，否则为 0 (失败/释放)
        sampled_gripper_action = np.where(probabilities > np.random.rand(*probabilities.shape), 0, 0.04)[:, None] # 闭合0 打开0.04
        state.info["current_gripper_action"] = sampled_gripper_action.squeeze()

        new_pos = np.concatenate([new_joint_pos, sampled_gripper_action], axis=-1)

        # step action
        cliped_new_pos = np.clip(new_pos, self.robot_joint_pos_min_limit, self.robot_joint_pos_max_limit, dtype=np.float32) # clip new pos to limit
        self._actuator_ctrl(state.data, cliped_new_pos)

        return state

    def update_state(self, state: NpEnvState):
        # compute obs
        obs = self._compute_observation(state.data, state.info)
        # compute truncated
        truncated = self._check_termination(state)

        # compute reward
        reward = self._compute_reward(state, truncated)

        state.obs = obs
        state.reward = reward
        state.terminated = truncated

        self.count += 1
        # if sum(state.terminated) > 0 and self.count < 200 and self.count > 150:
        #     # 查看成功率
        #     print("测试信息：")
        #     open_dist = self.drawer_top_joint.get_dof_pos(state.data).squeeze()
        #     res = open_dist > 0.20
        #     print(f"***环境数量{state.data.shape[0]}, 成功次数{sum(res)}, 成功率{sum(res)/state.data.shape[0] * 100}%***")

        return state

    def reset(self, data: mtx.SceneData):
        num_reset = data.shape[0]


        noise_pos = np.random.uniform(
           -self._cfg.control_config.joint_pos_reset_noise,
            self._cfg.control_config.joint_pos_reset_noise,
            (num_reset, self._num_dof_pos),
        )

        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1)) + noise_pos # 一定的噪声 -0.125, 0.125
        data.reset(self._model)
        data.set_dof_vel(np.zeros((num_reset, 13), dtype=np.float32)) # 包括机器人和柜子
        data.set_dof_pos(np.concatenate([dof_pos, np.zeros((num_reset, 4), dtype=np.float32)], axis=-1), self._model)
        self._model.forward_kinematic(data)

        info = {
            "current_actions": np.zeros((num_reset, self._action_dim), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._action_dim), dtype=np.float32),
            "phase2_mask": np.zeros(num_reset, dtype=bool), # 一维
            "current_gripper_action": np.zeros(num_reset, dtype=np.float32), # 一维
        }
        obs = self._compute_observation(data, info)
        return obs, info


    def _compute_observation(self, data: mtx.SceneData, info: dict):
        num_envs = data.shape[0]

        # dof_pos： (num_envs, 8) 取值范围: [-1 ~ 1]
        dof_pos = self.get_robot_joint_pos(data) # shape: (num_envs, 8)
        dof_pos_rel = self._get_robot_joint_pos_rel(dof_pos)[:, :self._action_dim]
        dof_lower_limits = np.tile(self.cfg.control_config.min_pos, (num_envs, 1))
        dof_upper_limits = np.tile(self.cfg.control_config.max_pos, (num_envs, 1))

        dof_pos_scaled = (
            2.0
            * dof_pos_rel
            / (dof_upper_limits - dof_lower_limits)
            - 1.0
        )
        # relative vel： (num_envs, 8) 取值范围大致为(-pi ~ pi) / 2 (除以2是稍微小一点)
        dof_vel = self.get_robot_joint_vel(data)
        dof_vel_rel = self._get_robot_joint_vel_rel(dof_vel)[:, :self._action_dim] / 2

        # relative orientation： (num_envs, 1)
        robot_grasp_pose = self.gripper_tcp.get_pose(data)
        drawer_grasp_pose = self.drawer_top_handle.get_pose(data)
        to_target = drawer_grasp_pose - robot_grasp_pose

        # cabinet joint
        drawer_top_joint_pos = self.drawer_top_joint.get_dof_pos(data) # shape: (num_envs, 1)
        drawer_top_joint_vel = self.drawer_top_joint.get_dof_vel(data) # shape: (num_envs, 1)

        obs = np.concatenate([dof_pos_scaled, dof_vel_rel, to_target, drawer_top_joint_pos, drawer_top_joint_vel], axis=-1)

        assert obs.shape == (num_envs, self._obs_dim)
        assert not np.isnan(obs).any(), "obs contain nan"
        return np.clip(obs, -5, 5)

    def _compute_reward(self, state: NpEnvState, truncated: np.ndarray):
        robot_grasp_pose = self.gripper_tcp.get_pose(state.data)
        drawer_grasp_pose = self.drawer_top_handle.get_pose(state.data)

        gripper_drawer_dist = np.linalg.norm(drawer_grasp_pose[:, :3] - robot_grasp_pose[:, :3], axis=-1)

        ## distance reward
        std =  0.1
        dist_reward = 1 - np.tanh(gripper_drawer_dist / std)
        dist_reward *= 10

        ## matching orientation reward
        quat_reward = quaternion_rotation_reward_np(robot_grasp_pose[:, -4:], drawer_grasp_pose[:, -4:])

        ## close gripper reward
        # 夹爪小于0.025时，关闭夹爪 奖励
        # 夹爪大于0.025时，关闭夹爪 惩罚
        # 夹爪大于0.025 或者 小于0.025时，张开夹爪  不奖励
        open_gripper= np.where(gripper_drawer_dist < 0.025, 100.0, -20) * (0.04-state.info["current_gripper_action"]) # dist_reward * 0 or 0.04

        ## open drawer reward
        open_dist = self.drawer_top_joint.get_dof_pos(state.data).squeeze()
        open_dist = np.clip(open_dist, 0, 1)
        open_reward = (np.exp(open_dist)-1) * 20

        wrong_open = np.logical_and(open_dist > 0 , gripper_drawer_dist > 0.03) # 抽屉开了并且夹爪不在把手上
        open_reward = np.bitwise_not(wrong_open) * open_reward            # 撞开的不给奖励 # 增大模型mjcf的阻力后，无法撞开


        ##################### 惩罚项 #####################
        ## action惩罚，
        ## joitn_vel惩罚，不过有的时候，部分关节还是转动大的，但是有的关节转动小
        action_penalty = np.sum(np.square(state.info["current_actions"] - state.info["last_actions"]), axis=-1)
        joint_vel_penalty = np.sum(np.square(state.data.dof_vel[:, :self._action_dim]), axis=-1)

        ## finger position penalty
        lfinger_dist = self.left_finger_pad.get_pose(state.data)[:, 2] - drawer_grasp_pose[:, 2]
        rfinger_dist = drawer_grasp_pose[:, 2] - self.right_finger_pad.get_pose(state.data)[:, 2]
        finger_dist_penalty = np.zeros_like(lfinger_dist)
        finger_dist_penalty += np.where(lfinger_dist < 0, lfinger_dist, np.zeros_like(lfinger_dist))
        finger_dist_penalty += np.where(rfinger_dist < 0, rfinger_dist, np.zeros_like(rfinger_dist))

        ##################### 系数变化 #####################

        ## action penalty rate
        if self.count < 12000:
            action_penalty_rate = 1e-4 * 10
            joint_vel_penalty_rate = 0 * 10 # 刚开始要很小
        else:
            action_penalty_rate = 2e-4 * 10
            joint_vel_penalty_rate = 2e-8 * 10

        ##################### 奖励计算 #####################


        step2_reward = dist_reward + quat_reward + open_gripper + open_reward  +  finger_dist_penalty

        # 奖励
        reward = step2_reward - action_penalty_rate * action_penalty - joint_vel_penalty_rate * joint_vel_penalty

        # 截断处理
        reward = np.where(truncated, reward-np.array(10.0), reward)

        # if self.count % 50 == 0:
        #     print()
        #     print()
        #     print("  gripper_drawer_dist:             ", gripper_drawer_dist)
        #     print("gripper_drawer_dist_reward:        ", dist_reward)
        #     print("  gripper_action                   ", state.info["current_gripper_action"])
        #     print("quat_reward:                      ", quat_reward)
        #     print("open_gripper:                      ", open_gripper)
        #     print("open_reward:                       ", open_reward)
        #     print("2.0 * finger_dist_penalty:         ",  finger_dist_penalty)
        #     print("action_penalty_rate * action_penalty:", action_penalty_rate * action_penalty)
        #     print("joint_vel_penalty_rate * joint_vel_penalty:", joint_vel_penalty_rate * joint_vel_penalty)
        #     print("reward.mean(axis=-1):               ", reward.mean(axis=-1))

        return reward



    def _check_termination(self, state: NpEnvState):
        # 超时截断
        truncated = state.info["steps"] >= self._cfg.max_episode_steps

        # 检查是否机械臂往前伸太远导致碰撞
        robot_grasp_pos_x = self.gripper_tcp.get_pose(state.data)[:, 0]
        drawer_grasp_pos_x = self.drawer_top_handle.get_pose(state.data)[:, 0]
        truncated = np.logical_or(truncated, robot_grasp_pos_x - drawer_grasp_pos_x < -0.03)

        # 检查关节速度不能超过阈值5弧度每秒
        joint_vel = self.get_robot_joint_vel(state.data)
        truncated = np.logical_or(truncated, np.abs(joint_vel).max(axis=-1) > 5)
        return truncated


    def _actuator_ctrl(self, data: mtx.SceneModel, value: np.ndarray):
        for i in range(self._action_dim): # 8个是actuator
            actuator = self._robot_actuators[i]
            actuator.set_ctrl(data, np.ascontiguousarray(value[:, i]))

    def get_robot_joint_pos(self, data: mtx.SceneModel):
        return self.robot.get_joint_dof_pos(data)[:, :self._num_dof_pos]

    def get_robot_joint_vel(self, data: mtx.SceneModel):
        return self.robot.get_joint_dof_vel(data)[:, :self._num_dof_pos]

    def _get_robot_joint_pos_rel(self, dof_pos: np.ndarray):
        return dof_pos - self.robot_default_joint_pos

    def _get_robot_joint_vel_rel(self, dof_vel: np.ndarray):
        return dof_vel - self._init_dof_vel
