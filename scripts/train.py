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


import logging  # 标准库日志模块，用于输出训练期间的调试与信息日志

from absl import app, flags  # absl 提供命令行解析与应用入口封装
from skrl import config  # skrl 框架的全局配置对象，可设置后端等参数

from motrix_rl import utils  # 本项目的工具集，包含设备支持探测等功能

# logger: 本文件使用的日志记录器，名字为当前模块名，受全局日志配置控制
logger = logging.getLogger(__name__)

# 训练阶段的命令行参数定义（absl flags），每个都是全局单例，.value 访问实际值，.present 表示是否显式传参
# _ENV: 要训练的环境名称，需在 motrix_envs.registry 注册；默认 cartpole 用于示例
_ENV = flags.DEFINE_string("env", "cartpole", "The env to train")
# _SIM_BACKEND: 物理仿真后端（如 np）；None 时自动选择该环境的第一个支持后端
_SIM_BACKEND = flags.DEFINE_string(
    "sim-backend",
    None,
    "The simulation backend to use.(If not specified, it will be choosen automatically)",
)

# ============================================================================
# 本文件负责提供训练入口脚本。主要流程：
# 1. 解析命令行参数，确定环境、并行数量、随机种子等配置；
# 2. 根据当前机器的硬件与安装情况，推断最佳的训练后端（jax 或 torch）；
# 3. 创建对应后端的 PPO Trainer 实例；
# 4. 调用 Trainer.train() 启动强化学习训练循环。
# 该脚本被直接运行时（python scripts/train.py 或 uv run ...）会触发 main 函数。
# ============================================================================
# _NUM_ENVS: 并行环境数量（向量化采样），直接影响每步采样吞吐与显存占用
_NUM_ENVS = flags.DEFINE_integer("num-envs", 2048, "Number of envs to train")
# _RENDER: 是否在训练时开启渲染，默认 False 以节省算力；True 时会打开可视化窗口
_RENDER = flags.DEFINE_bool("render", False, "Render the env")
# _TRAIN_BACKEND: 强化学习计算后端，支持 jax/torch；若未显式传参，后面会根据设备自动选择
_TRAIN_BACKEND = flags.DEFINE_string("train-backend", "jax", "The learning backend. (jax/torch)")
# _SEED: 固定随机种子，方便复现；当 _RAND_SEED 为 True 时会被覆盖
_SEED = flags.DEFINE_integer("seed", None, "Random seed for reproducibility")
# _RAND_SEED: 若 True 则强制使用随机种子（不固定），提升多样性但牺牲复现性
_RAND_SEED = flags.DEFINE_bool("rand-seed", False, "Generate random seed")
# _PRETRAINED: optional checkpoint path loaded before training starts for transfer learning.
_PRETRAINED = flags.DEFINE_string("pretrained", None, "Path to a checkpoint to load before training")


def get_train_backend(supports: utils.DeviceSupports):
    """根据设备支持情况返回最合适的训练后端。"""
    # supports.jax: bool，表示是否安装并可导入 jax；supports.jax_gpu 表示是否可用 GPU 加速
    # supports.torch: bool，表示是否安装并可导入 torch；supports.torch_gpu 表示是否可用 GPU 加速
    if supports.jax and supports.jax_gpu:
        return "jax"
    elif supports.torch and supports.torch_gpu:
        return "torch"
    elif supports.jax:
        return "jax"
    elif supports.torch:
        return "torch"
    else:
        raise Exception("neither jax nor torch not avaliable on the device.")


def main(argv):
    """训练脚本主入口：参数解析、后端选择与训练启动由此完成。"""
    # device_supports: DeviceSupports 实例，包含是否支持 torch/jax 以及是否有 GPU
    device_supports = utils.get_device_supports()
    logger.info(device_supports)

    # 读取命令行配置
    env_name = _ENV.value  # 环境名称字符串，传递给 Trainer
    enable_render = _RENDER.value  # bool，训练时是否开启渲染
    pretrained_path = _PRETRAINED.value

    # rl_override 用于向 Trainer 传递少量覆盖参数（目前 num_envs 与 seed）
    rl_override = {}  # 字典，键值最终被 Trainer 内部读取并应用到 RL 配置
    if _NUM_ENVS.present:
        rl_override["num_envs"] = _NUM_ENVS.value  # 覆盖并行环境数量

    if _RAND_SEED.value:
        rl_override["seed"] = None  # 使用随机种子，训练过程不可复现
    elif _SEED.present:
        rl_override["seed"] = _SEED.value  # 固定种子，确保复现性

    # 选择仿真后端（env）和学习后端（RL）
    sim_backend = _SIM_BACKEND.value  # 物理仿真后端字符串，None 表示自动选择
    train_backend = "jax"  # 训练后端默认值，后续可能被自动/手动覆盖
    if not _TRAIN_BACKEND.present:
        train_backend = get_train_backend(device_supports)  # 未指定时按设备优先级自动选择
    else:
        train_backend = _TRAIN_BACKEND.value  # 用户显式指定

    # 根据后端加载对应的 PPO Trainer 实现
    trainer = None  # Assigned below based on the selected backend.
    if train_backend == "jax":
        from motrix_rl.skrl.jax.train import ppo

        config.jax.backend = "jax"  # 可切换为 "numpy" 以纯 CPU 运行（降级模式）
        # trainer: 来自 motrix_rl.skrl.jax.train.ppo 的 Trainer 类实例，封装 PPO 训练循环
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)

    elif train_backend == "torch":
        from motrix_rl.skrl.torch.train import ppo

        config.torch.backend = "torch"  # 指定 skrl 使用 torch 后端
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)
    else:
        raise Exception(f"Unknown train backend: {train_backend}")

    # 启动训练循环
    trainer.train(pretrained=pretrained_path)


if __name__ == "__main__":
    app.run(main)
