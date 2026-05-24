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

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.skrl.cfg import PPOCfg


class basic:
    @rlcfg("cartpole")
    @dataclass
    class CartPolePPO(PPOCfg):
        max_env_steps: int = 10_000_000
        check_point_interval: int = 500

        # Override PPO configuration
        policy_hidden_layer_sizes: tuple[int, ...] = (32, 32)
        value_hidden_layer_sizes: tuple[int, ...] = (32, 32)
        rollouts: int = 32
        learning_epochs: int = 5
        mini_batches: int = 4

    @rlcfg("dm-walker", backend="jax")
    @rlcfg("dm-stander", backend="jax")
    @rlcfg("dm-runner", backend="jax")
    @dataclass
    class WalkerPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 4

    @rlcfg("dm-stander", backend="torch")
    @rlcfg("dm-walker", backend="torch")
    @dataclass
    class WalkerPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 32

    @rlcfg("dm-runner", backend="torch")
    @dataclass
    class RunnerPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 2
        mini_batches: int = 32


class locomotion:
    @rlcfg("go1-flat-terrain-walk")
    @dataclass
    class Go1WalkPPO(PPOCfg):
        """
        Go1 Walk RL config
        """

        seed: int = 42
        share_policy_value_features: bool = False
        max_env_steps: int = 1024 * 60000
        num_envs: int = 2048

        # Override PPO configuration
        rollouts: int = 24
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        learning_epochs: int = 5
        mini_batches: int = 3
        learning_rate: float = 3e-4


class navigation:
    @rlcfg("anymal-c-flat-terrain-nav")
    @dataclass
    class AnymalCNavPPO(PPOCfg):
        """
        Anymal C Navigation RL config
        """

        seed: int = 42
        share_policy_value_features: bool = False
        max_env_steps: int = 1024 * 500_000  # About 5 亿步
        num_envs: int = 2048

        # Override PPO configuration
        rollouts: int = 24
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        learning_epochs: int = 5
        mini_batches: int = 4
        learning_rate: float = 3e-4
