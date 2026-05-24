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

import logging  # 标准库日志模块，用于打印推理/演示阶段的日志
from pathlib import Path  # 处理文件与目录路径，便于跨平台

from absl import app, flags  # absl 提供命令行参数解析和入口包装
from skrl import config  # skrl 框架的全局配置对象，用于设置后端等

from motrix_rl import utils  # 本项目工具库，包含设备能力探测等
from motrix_rl.skrl import get_log_dir  # 返回 runs 日志目录路径的辅助函数

# logger: 当前模块的日志记录器，输出取决于全局日志配置；
# 通过调整全局 logging 设置可以控制终端或文件中的日志级别
logger = logging.getLogger(__name__)

# -------------------------
# 推理/演示阶段命令行参数说明：
# 每个 flag 都可通过命令行传入并影响推理行为，.value 访问最终值，.present 判断是否显式设置。
# -------------------------

# _ENV: 需要加载的环境 ID；必须在 motrix_envs.registry 完成注册，默认值 cartpole 方便快速测试
_ENV = flags.DEFINE_string("env", "cartpole", "推理或演示所用的环境名称")

# _SIM_BACKEND: 选择具体的仿真后端（如 "np"），若不指定则自动选择环境支持的首个后端
_SIM_BACKEND = flags.DEFINE_string(
    "sim-backend",
    None,
    "使用的仿真后端（未指定则自动选择可用后端）",
)

# _POLICY: 提供明确的策略文件路径；若为空则脚本会在 runs 目录中按照规则搜索最优/最新的权重
_POLICY = flags.DEFINE_string("policy", None, "要加载的策略文件路径")

# _NUM_ENVS: 推理阶段创建的并行环境数量；与训练相比通常可设较小以节省资源
_NUM_ENVS = flags.DEFINE_integer("num-envs", 2048, "推理或演示时的并行环境数量")

# _SEED: 固定随机种子，确保多次推理获得一致的轨迹；当 _RAND_SEED 为 True 时该值被忽略
_SEED = flags.DEFINE_integer("seed", None, "固定随机种子以便复现")

# _RAND_SEED: 开启后忽略 --seed，转而使用随机种子；适合观察策略在不同扰动下的表现
_RAND_SEED = flags.DEFINE_bool("rand-seed", False, "是否使用随机种子（非固定）")


def get_inference_backend(policy_path: str):
    """根据策略文件后缀自动推断推理使用的后端。

    对于本项目：
    - 以 .pt 结尾的文件默认对应 torch 保存的权重；
    - 以 .pickle 结尾的文件默认对应 jax 保存的参数；
    若不满足上述格式，则抛出异常提醒用户检查文件类型。

    参数:
        policy_path: str，策略权重的完整路径。

    返回:
        字符串 "torch" 或 "jax"，用于进一步选择 Trainer 实现。
    """
    # policy_path: str，用户显式传入或自动推断得到的 checkpoint 路径
    if policy_path.endswith(".pt"):
        return "torch"
    if policy_path.endswith(".pickle"):
        return "jax"
    else:
        raise Exception(f"Unknown policy format: {policy_path}")


def find_best_policy(env_name: str) -> str:
    """在 runs 目录中自动查找目标环境的策略权重。

    查找规则：
    1. 首先定位 runs/<env_name>/ 下最近一次训练 run（依据修改时间排序）；
    2. 优先返回 checkpoints 目录下名为 best_agent.* 的最佳权重；
    3. 若没有 best_agent.*，则从 agent_*.pt 或 agent_*.pickle 中选择时间步数最大的一个；
    4. 任一步骤失败均抛出 FileNotFoundError，并提示用户手动指定策略。

    参数:
        env_name: str，环境名称，需与训练时使用的一致。

    返回:
        策略权重的绝对路径字符串。

    异常:
        FileNotFoundError: 未找到训练记录或策略文件时抛出。
    """
    # env_dir: Path 对象，指向 runs 目录下对应环境的输出路径
    env_dir = Path(get_log_dir(env_name))

    if not env_dir.exists():
        raise FileNotFoundError(f"No training results found for environment '{env_name}' in {env_dir}")

    # training_runs: List[Path]，收集所有子目录（每个子目录对应一次训练运行）
    # 命名约定通常为 YY-MM-DD_HH-MM-SS-XXXXXX_PPO，具体值由训练脚本生成
    training_runs = [d for d in env_dir.iterdir() if d.is_dir()]

    if not training_runs:
        raise FileNotFoundError(f"No training runs found for environment '{env_name}'")

    # latest_run: Path，依据最后修改时间排序后得到最新的训练运行
    latest_run = max(training_runs, key=lambda x: x.stat().st_mtime)
    # checkpoints_dir: Path，指向最新 run 中的 checkpoints 子目录
    checkpoints_dir = latest_run / "checkpoints"

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found in {latest_run}")

    # best_files: List[Path]，匹配 best_agent.* 文件，通常保存最高评估分数的策略
    best_files = list(checkpoints_dir.glob("best_agent.*"))

    if best_files:
        # Return the first best_agent file found (there should only be one)
        return str(best_files[0])

    # checkpoint_files: List[Path]，包含所有 agent_*.pt/pickle，需从中挑选时间步最大的一个
    checkpoint_files = list(checkpoints_dir.glob("agent_*.pt")) + list(checkpoints_dir.glob("agent_*.pickle"))

    if not checkpoint_files:
        raise FileNotFoundError(f"No policy files found in {checkpoints_dir}")

    # extract_timestep: 辅助函数，从文件名中解析出时间步编号，用于比较并挑选最新模型
    def extract_timestep(filename):
        # 期望文件名格式为 agent_{timestep}.ext
        stem = Path(filename).stem  # 取出不包含扩展名的部分
        parts = stem.split("_")  # 通过下划线拆分，第二段应为数字时间步
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return 0
        return 0

    latest_checkpoint = max(checkpoint_files, key=extract_timestep)
    return str(latest_checkpoint)


def main(argv):
    """推理脚本主入口，负责串联权重定位、后端选择与播放流程。

    主要步骤：
    1. 通过 utils.get_device_supports() 检测本机对 jax/torch 的支持以及是否具备 GPU；
    2. 读取命令行参数，建立 rl_override 用于给 Trainer 传递 play_num_envs 与 seed；
    3. 若用户未指定策略文件，则自动调用 find_best_policy 搜索最新/最优权重；
    4. 根据策略文件后缀判定推理后端，加载对应的 Trainer；
    5. 调用 trainer.play(policy_path) 执行推理或演示。

    参数:
        argv: 来自 absl.app.run 的命令行参数列表，通常无需直接使用。
    """
    device_supports = utils.get_device_supports()  # 探测硬件能力（jax/torch 与 GPU 情况）
    logger.info(device_supports)
    env_name = _ENV.value  # env_name: str，目标环境名称，需要与训练时保持一致
    enable_render = True  # enable_render: bool，此处默认开启渲染，可视化策略行为

    rl_override = {}  # rl_override: dict[str, Any]，提供给 Trainer.play 的覆盖配置

    # 仅在命令行显式传入 --num-envs 时才覆盖默认的 play_num_envs
    if _NUM_ENVS.present:
        rl_override["play_num_envs"] = _NUM_ENVS.value

    # 种子选择逻辑：rand-seed 优先，其次是显式 seed，最后采用默认配置
    if _RAND_SEED.value:
        rl_override["seed"] = None
    elif _SEED.present:
        rl_override["seed"] = _SEED.value

    sim_backend = _SIM_BACKEND.value  # sim_backend: Optional[str]，仿真后端

    # 确定策略文件路径：若用户提供 --policy，则直接使用；否则自动搜索
    if _POLICY.present:
        policy_path = _POLICY.value
        logger.info(f"Using specified policy: {policy_path}")
    else:
        try:
            policy_path = find_best_policy(env_name)
            logger.info(f"Auto-discovered best policy: {policy_path}")
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            logger.error("Please specify a policy using --policy flag or train a model first")
            return

    # 根据策略文件扩展名决定推理后端（torch 或 jax）
    backend = get_inference_backend(policy_path)

    if backend == "jax":
        assert device_supports.jax, "jax is not avaliable on your device "  # 若 jax 不可用则中止
        from motrix_rl.skrl.jax.train import ppo  # 导入 jax 版本的 PPO Trainer

        config.jax.backend = "jax"  # 若希望强制使用 CPU，可修改为 "numpy"
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)
        trainer.play(policy_path)  # 加载策略并执行推理/演示

    elif backend == "torch":
        assert device_supports.torch, "torch is not avaliable on your device"  # 若 torch 不可用则中止
        from motrix_rl.skrl.torch.train import ppo  # 导入 torch 版本的 PPO Trainer

        config.torch.backend = "torch"
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)
        trainer.play(policy_path)


if __name__ == "__main__":
    app.run(main)
