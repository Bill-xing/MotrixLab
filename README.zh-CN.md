**语言**: [English](README.md) | [简体中文](README.zh-CN.md)

# MotrixLab

![GitHub License](https://img.shields.io/github/license/Motphys/MotrixLab)
![Python Version](https://img.shields.io/badge/python-3.10-blue)

`MotrixLab` 是一个基于 [MotrixSim](https://github.com/Motphys/motrixsim-docs) 仿真引擎的强化学习框架，专为机器人仿真和训练设计。该项目提供了一个完整的强化学习开发平台，集成了多种仿真环境、任务资产和训练框架。

## 项目概述

该项目分为两个核心部分：

-   **motrix_envs**: 基于 MotrixSim 构建的各种 RL 仿真环境，定义 observation、action、reward、场景资产和任务注册。与具体的 RL 框架无关，目前支持 MotrixSim 的 NumPy 仿真后端
-   **motrix_rl**: 集成 RL 框架，并使用 motrix_envs 里的环境参数进行训练。目前支持 SKRL 框架的 PPO 算法，以及 JAX 和 PyTorch 训练后端

> 文档地址：https://motrixlab.readthedocs.io

## 主要特性

-   **统一接口**: 提供简洁统一的强化学习训练和评估接口
-   **多后端支持**: 支持 JAX 和 PyTorch 训练后端，可根据硬件环境灵活选择
-   **丰富环境**: 包含基础控制、四足运动、四足导航和 Franka 操作等多种机器人仿真环境
-   **崎岖地形导航**: 提供基于 MuJoCo heightfield 的 ANYmal-C 崎岖地形导航任务，包含智能目标采样、地形感知重置和额外稳定性奖励
-   **迁移学习**: 通用训练脚本支持通过 `--pretrained` 在训练前加载 JAX PPO checkpoint
-   **高性能仿真**: 基于 MotrixSim 的高性能物理仿真引擎
-   **可视化训练**: 支持实时渲染和训练过程可视化

## 可用环境

| 类别 | 环境 ID | 说明 |
| --- | --- | --- |
| 基础控制 | `cartpole` | 倒立摆平衡任务 |
| 基础控制 | `dm-walker`, `dm-stander`, `dm-runner` | DeepMind 风格 walker 任务 |
| 四足运动 | `go1-flat-terrain-walk` | Unitree Go1 平地行走 |
| 四足运动 | `go1-rough-terrain-walk` | Unitree Go1 崎岖地形行走 |
| 四足运动 | `go1-stairs-terrain-walk` | Unitree Go1 楼梯地形行走 |
| 四足运动 | `go2-flat-terrain-walk` | Unitree Go2 平地行走 |
| 四足运动 | `anymal_c_locomotion_flat` | ANYmal-C 平地运动 |
| 四足导航 | `anymal-c-flat-terrain-nav`, `anymal_c_navigation_flat` | ANYmal-C 平地位置和朝向导航 |
| 四足导航 | `anymal_c_navigation_rough` | ANYmal-C 崎岖地形位置和朝向导航 |
| 机械臂操作 | `franka_lift_cube` | Franka 方块抓取抬升 |
| 机械臂操作 | `franka_open_cabinet` | Franka 柜门开启 |

## 🚀 快速开始

> 以下示例使用了 Python 项目管理工具：[UV](https://docs.astral.sh/uv/)
>
> 在开始之前，请先[安装](https://docs.astral.sh/uv/getting-started/installation/)该工具。

### 克隆仓库

```bash
git clone https://github.com/Motphys/MotrixLab

cd MotrixLab

git lfs pull
```

### 安装依赖

安装全部依赖：

```bash
uv sync --all-packages --all-extras
```

SKRL 框架支持 JAX(Flax)或 PyTorch 作为训练后端，您也可以根据自己的设备环境，选择只安装其中一种训练后端：

安装 JAX 作为训练后端（仅支持 Linux 平台）：

```bash
uv sync --all-packages --extra skrl-jax
```

安装 PyTorch 作为训练后端：

```bash
uv sync --all-packages --extra skrl-torch
```

## 🎯 使用指南

### 环境可视化

查看环境而不执行训练：

```bash
uv run scripts/view.py --env cartpole
```

查看 ANYmal-C 崎岖地形导航任务：

```bash
uv run scripts/view.py --env anymal_c_navigation_rough --num-envs 1
```

### 训练模型

```bash
uv run scripts/train.py --env cartpole
```

训练结果会保存在 `runs/{env-name}/` 目录下。

通过 TensorBoard 查看训练数据：

```bash
uv run tensorboard --logdir runs/{env-name}
```

训练 ANYmal-C 崎岖地形导航任务：

```bash
uv run scripts/train.py --env anymal_c_navigation_rough --num-envs 2048 --train-backend jax
```

从 JAX PPO checkpoint 继续训练：

```bash
uv run scripts/train.py --env anymal_c_navigation_rough --pretrained path/to/checkpoint.pickle
```

### 模型推理

```
uv run scripts/play.py --env cartpole
```

加载指定策略文件：

```bash
uv run scripts/play.py --env anymal_c_navigation_rough --policy path/to/best_agent.pickle --num-envs 64
```

## ANYmal-C 崎岖地形导航

`anymal_c_navigation_rough` 环境将 Isaac Lab 的 `isaac-navigation-flat-anymal-c-v0` 任务迁移到 MotrixLab，并将平面地形替换为复杂 heightfield 地形。

![ANYmal-C 崎岖地形导航演示](docs/source/_static/images/demo/anymal_c_rough_navigation.gif)

-   **动作空间**: `Box(-1.0, 1.0, (12,), float32)`，对应 ANYmal-C 12 个关节的位置控制命令
-   **观测空间**: `Box(-inf, inf, (54,), float32)`
-   **地形**: 由 `heightmap.png` 驱动的 40 m x 40 m MuJoCo heightfield，世界高度范围为 -1.0 m 到 1.5 m
-   **任务目标**: 导航到采样的 XY 目标位置和目标 yaw，并在地形表面稳定停止
-   **回合长度**: 1500 个控制步，默认 100 Hz 控制频率下为 15 秒
-   **重置和目标采样**: 基于地形高度查询、高度差限制、坡度检查和安全回退生成机器人初始位置与目标位置
-   **奖励函数**: 在平地导航奖励基础上增加垂直速度稳定性、机身姿态稳定性和足部接触奖励
-   **终止条件**: 包含物理失败检测，以及 19 m 地形边界检测

更多使用方式请参考[用户文档](https://motrixlab.readthedocs.io)

## 📬 联系方式

有问题或建议？欢迎通过以下方式联系我们：

-   GitHub Issues: [提交问题](https://github.com/Motphys/MotrixLab/issues)
-   Discussions: [加入讨论](https://github.com/Motphys/MotrixLab/discussions)
