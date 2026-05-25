import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import FrankaLiftCubeEnvCfg

np.set_printoptions(precision=4)

@registry.env("franka_lift_cube", "np")
class FrankaLiftCubeEnv(NpEnv):
    _cfg: FrankaLiftCubeEnvCfg

    def __init__(self, cfg: FrankaLiftCubeEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self.default_joint_pos = self._cfg.init_state.default_joint_pos

        self._action_dim = 8
        self._obs_dim = 36 # 9 + 9 + 3 + 7 + 8
        self._action_space = gym.spaces.Box(-np.inf, np.inf, (self._action_dim,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (self._obs_dim,), dtype=np.float32)

        self._num_dof_pos = 9 # self._model.num_dof_pos # 9
        self._num_dof_vel = 9 # self._model.num_dof_vel # 9
        self._init_dof_pos = self.default_joint_pos
        self._init_dof_vel = np.zeros(self._num_dof_vel, dtype=np.float32)


        # 一些属性
        self._cube = self._model.get_geom("cube")
        self._body = self._model.get_body("link0")


        self.hand = self._model.get_site("gripper")

        self.joint_pos_min_limit = self._cfg.control_config.min_pos
        self.joint_pos_max_limit = self._cfg.control_config.max_pos

        # step统计
        self.count = 0


    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions

        # no gripper
        action_scale = 0.5
        old_joint_pos = np.tile(self._init_dof_pos[:self._action_dim-1] , (self._num_envs, 1))
        new_joint_pos = actions[:, :self._action_dim-1] * action_scale  + old_joint_pos # action as offset

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
        cliped_new_pos = np.clip(new_pos, self.joint_pos_min_limit, self.joint_pos_max_limit, dtype=np.float32) # clip new pos to limit

        state.data.actuator_ctrls = cliped_new_pos
        return state

    def update_state(self, state: NpEnvState):

        # compute observation
        obs = self._compute_observation(state.data, state.info)

        # compute truncated
        truncated = self._check_termination(state)

        # compute reward
        reward = self._compute_reward(state)

        # set command visulizer
        self.set_command_visulizer(state.data, state.info["commands"])

        state.obs = obs
        state.reward = reward
        state.terminated = truncated

        self.count += 1

        return state


    def reset(self, data: mtx.SceneData):
        num_reset = data.shape[0]

        # 机械臂初始关节角度噪声
        noise_pos = np.random.uniform(
            -self._cfg.init_state.joint_pos_reset_noise,
            self._cfg.init_state.joint_pos_reset_noise,
            self._num_dof_pos
        )
        robot_dof_pos = self._init_dof_pos + noise_pos

        # 对 cube 的初始位置 进行域随机化
        x_low, x_high = self._cfg.init_state.cube_pos_x_reset_noise
        y_low, y_high = self._cfg.init_state.cube_pos_y_reset_noise
        pos_x_delta = np.random.uniform(x_low, x_high)
        pos_y_delta = np.random.uniform(y_low, y_high)

        cube_dof_pose = np.array([pos_x_delta, pos_y_delta, 0.05, 1, 0, 0, 0], dtype=np.float32)

        command = self._generated_commands(num_reset) # 目标位置 用一个坐标系来可视化

        scene_dof_pos = np.concatenate([robot_dof_pos, cube_dof_pose])# 新增的cube freejoint dof pos
        scene_dof_pos = np.tile(scene_dof_pos, (num_reset, 1))
        scene_dof_pos = np.concatenate([scene_dof_pos, command, np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (num_reset, 1))], axis=-1) # 新增的目标位置

        scene_dof_vel = np.concatenate([self._init_dof_vel, np.zeros(6+6, dtype=np.float32)]) # 6 + 6 是cube和目标位置坐标系两个freejoint的dof vel
        scene_dof_vel = np.tile(scene_dof_vel, (num_reset, 1))


        # 重置
        data.reset(self._model)
        data.set_dof_vel(scene_dof_vel)
        data.set_dof_pos(scene_dof_pos, self._model)
        self._model.forward_kinematic(data)


        cube_pos = np.array([0.5 + pos_x_delta, 0 + pos_y_delta, 0.02])
        info = {
            "current_actions": np.zeros((num_reset, self._action_dim), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._action_dim), dtype=np.float32),
            "commands": command, #
            "current_gripper_action": np.zeros(num_reset, dtype=np.float32), # 一维
            "command_cube_max_length": np.linalg.norm(cube_pos - command, axis=-1)
        }

        obs = self._compute_observation(data, info)
        return obs, info


    def _compute_observation(self, data: mtx.SceneData, info: dict):

        dof_pos = self.get_dof_pos(data) # 被update_state调用shape为(self.num_envs, 9)；被reset调用则为(num_reset, 9)
        dof_vel = self.get_dof_vel(data)

        # 机械臂关节相对于初始状态的角度和速度
        dof_pos_rel = self._get_joint_pos_rel(dof_pos)
        dof_vel_rel = self._get_joint_vel_rel(dof_vel)

        # 物块的位置和姿态
        object_pick_pose = self._cube.get_pose(data)

        # 物块的目标位置
        object_lift_pos = info["commands"]

        # 机械臂上一个动作
        last_actions = info["current_actions"]

        # 拼接所有观测
        obs = np.concatenate([dof_pos_rel, dof_vel_rel, object_pick_pose, object_lift_pos, last_actions], axis=-1)

        assert obs.shape == (data.shape[0], self._obs_dim)
        assert not np.isnan(obs).any(), "obs contain nan"
        return obs.astype(np.float32)


    def _check_termination(self, state: NpEnvState):
        # 超时
        truncated = state.info["steps"] >= self._cfg.max_episode_steps

        # 物块从桌子上掉落
        cube_height = self._cube.get_pose(state.data)[:, 2]
        truncated = np.logical_or(truncated, cube_height < -0.05)

        # 检查关节速度不能过大 (这里设5弧度每秒)
        joint_vel = self.get_dof_vel(state.data)
        truncated = np.logical_or(truncated, np.abs(joint_vel).max(axis=-1) > 10)

        # 检查 物块 的速度
        cube_vel = self._cube.get_linear_velocity(state.data) # shape = (*data.shape, 3).
        truncated = np.logical_or(truncated, np.abs(cube_vel).max(axis=-1) > 10)
        return truncated


    def _compute_reward(self, state: NpEnvState):

        hand_pose = self.hand.get_pose(state.data)
        hand_pos, hand_quat = hand_pose[:, :3], hand_pose[:, 3:7]
        cube_pos = self._cube.get_pose(state.data)[:, :3]

        # 前往方块的奖励
        hand_cube_distance = np.linalg.norm(cube_pos - hand_pos, axis=-1)

        std =  0.1
        reach_reward = 1 - np.tanh(hand_cube_distance / std)

        # 举起方块的奖励
        lift_height = cube_pos[:, 2]
        minimal_height = 0.04
        lifted = lift_height > minimal_height

        # 前往目标位置的奖励
        object_command_dist = np.linalg.norm(cube_pos - state.info["commands"], axis=-1)
        ## 进度百分比奖励
        command_progress = object_command_dist / state.info["command_cube_max_length"]
        command_progress_reward = (1 - np.tanh(command_progress / 0.4))
        ## 靠近目标位置奖励
        command_tracking_reward = (1 - np.tanh(object_command_dist / 0.3)) * (lift_height > 0.04) * (hand_cube_distance < 0.02)
        ## 到达目标位置奖励
        command_reaching_reward = (1 - np.tanh(object_command_dist / 0.05)) * (object_command_dist < 0.3) * (hand_cube_distance < 0.02)

        # 惩罚项：动作改变量、机械臂速度改变量、机械臂关节改变量
        action_diff_sq = np.sum(np.square(state.info["current_actions"] - state.info["last_actions"]), axis=-1)

        joint_vel_sq = np.sum(np.square(self.get_dof_vel(state.data)[:, :self._num_dof_vel]), axis=1)
        joint_dof_sq = np.sum(np.square(self.get_dof_pos(state.data)[:, :self._num_dof_pos] - np.tile(self._init_dof_pos, (state.data.shape[0], 1))), axis=1)

        # 到达目标后静止奖励
        end_still_reward = (1 - np.tanh(joint_vel_sq / 0.4))* (object_command_dist < 0.04)

        # 奖励权重
        reach_weight = 1.
        lifted_weight = 10

        command_progress_weight = 100
        command_tracking_weight = 20
        command_reaching_weight = 220
        end_still_weight = 50

        if self.count < 10000:
            action_penalty_weight = 1e-4
            joint_vel_penalty_weight = 1e-4
        else:
            action_penalty_weight = 1e-1
            joint_vel_penalty_weight = 1e-1

        joint_dof_penalty_weight = 1e-2

        reward = reach_weight * reach_reward + \
            lifted_weight * lifted * (hand_cube_distance < 0.05) + \
            command_progress_weight * command_progress_reward * lifted + \
            command_tracking_weight * command_tracking_reward + \
            command_reaching_weight * command_reaching_reward + \
            - action_penalty_weight * action_diff_sq + \
            - joint_vel_penalty_weight * joint_vel_sq + \
            - joint_dof_penalty_weight * joint_dof_sq + \
            + end_still_weight * end_still_reward


        # if self.count % 50 == 0:
        #     print()
        #     print()
        #     print("hand_cube_distance:                          ", hand_cube_distance)
        #     print("reach_reward: 7                              --", reach_weight * reach_reward)
        #     print("lift_height:                                 ", lifted_weight * lifted * (hand_cube_distance < 0.05))

        #     print()
        #     print("object_command_dist:                        --", object_command_dist)
        #     print("command_progress:                            --", command_progress)
        #     print()
        #     print("command_progress_reward:                     --", command_progress_weight * command_progress_reward * lifted)
        #     print("command_tracking_reward:                     --", command_tracking_weight * command_tracking_reward)
        #     print("command_reaching_reward:                     --", command_reaching_weight * command_reaching_reward)
        #     print("action_penalty_rate * action_diff_sq:        --", action_penalty_weight * action_diff_sq)
        #     print("joint_vel_penalty_rate * joint_vel_sq:       --", joint_vel_penalty_weight * joint_vel_sq)
        #     print("joint_dof_penalty_weight * joint_dof_sq:     --", joint_dof_penalty_weight * joint_dof_sq)
        #     print("joint_vel_sq:                                --", joint_vel_sq)
        #     print("end_still_reward:                            --", end_still_weight * end_still_reward)

        #     print("reward.mean(axis=-1):                         ", reward.mean(axis=-1))

        return reward


    def get_dof_pos(self, data: mtx.SceneModel):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneModel):
        return self._body.get_joint_dof_vel(data)

    def _get_joint_pos_rel(self, dof_pos: np.ndarray):
        return dof_pos - self.default_joint_pos

    def _get_joint_vel_rel(self, dof_vel: np.ndarray):
        return dof_vel - self._init_dof_vel


    def _generated_commands(self, num_envs: int):
        # 命令是 cube应该到达的最后的位置
        x_low, x_high = self._cfg.command_config.target_pos_x
        y_low, y_high = self._cfg.command_config.target_pos_y
        z_low, z_high = self._cfg.command_config.target_pos_z

        # print("x_low, x_high: ", x_low, x_high)
        # print("y_low, y_high: ", y_low, y_high)
        # print("z_low, z_high: ", z_low, z_high)

        pos_x = np.random.uniform(x_low, x_high, num_envs)
        pos_y = np.random.uniform(y_low, y_high, num_envs)
        pos_z = np.random.uniform(z_low, z_high, num_envs)
        command_cube_target_pos = np.stack([pos_x, pos_y, pos_z], axis=-1)

        assert not np.isnan(command_cube_target_pos).any(), "command_cube_target_pos contain nan"
        return command_cube_target_pos

    def set_command_visulizer(self, data: mtx.SceneData, commands: np.ndarray):
        # 设置可视化坐标轴的位置
        # 获取所有环境的dof_pos
        all_dof_pos = data.dof_pos.copy()  # [num_envs, num_dof]
        # 可视化坐标轴x,y,z坐标的索引为 倒数第7个到倒数第5个
        all_dof_pos[:, -7:-4] = commands[:, :3]
        # 一次性设置所有环境的dof_pos
        data.set_dof_pos(all_dof_pos, self._model)
        # 必须调用forward_kinematic才能更新body的pose
        self._model.forward_kinematic(data)
