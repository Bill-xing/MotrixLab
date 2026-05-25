**Language**: [English](README.md) | [简体中文](README.zh-CN.md)

# MotrixLab

![GitHub License](https://img.shields.io/github/license/Motphys/MotrixLab)
![Python Version](https://img.shields.io/badge/python-3.10-blue)

`MotrixLab` is a reinforcement learning framework based on the [MotrixSim](https://github.com/Motphys/motrixsim-docs) simulation engine, designed specifically for robot simulation and training. This project provides a complete reinforcement learning development platform that integrates multiple simulation environments, task assets, and training frameworks.

## Project Overview

The project is divided into two core components:

-   **motrix_envs**: Various RL simulation environments built on MotrixSim, defining observation, action, reward, scene assets, and task registration. Framework-agnostic and currently supports MotrixSim's NumPy simulation backend
-   **motrix_rl**: Integrates RL frameworks and uses environment parameters from motrix_envs for training. Currently supports the SKRL framework's PPO algorithm with JAX and PyTorch training backends

> Documentation: https://motrixlab.readthedocs.io

## Key Features

-   **Unified Interface**: Provides a concise and unified reinforcement learning training and evaluation interface
-   **Multi-backend Support**: Supports JAX and PyTorch training backends, with flexible selection based on hardware environment
-   **Rich Environments**: Includes basic control, quadruped locomotion, quadruped navigation, and Franka manipulation tasks
-   **Rough-terrain Navigation**: Provides an ANYmal-C navigation task on a MuJoCo heightfield terrain with smart target sampling, terrain-aware reset, and additional stability rewards
-   **Transfer Learning**: Supports loading a pretrained JAX PPO checkpoint before training through the generic training script
-   **High-performance Simulation**: Built on MotrixSim's high-performance physics simulation engine
-   **Visual Training**: Supports real-time rendering and training process visualization

## Available Environments

| Category | Environment ID | Description |
| --- | --- | --- |
| Basic control | `cartpole` | Cart-pole balancing task |
| Basic control | `dm-walker`, `dm-stander`, `dm-runner` | DeepMind-style walker tasks |
| Locomotion | `go1-flat-terrain-walk` | Unitree Go1 flat-terrain walking |
| Locomotion | `go1-rough-terrain-walk` | Unitree Go1 rough-terrain walking |
| Locomotion | `go1-stairs-terrain-walk` | Unitree Go1 stairs-terrain walking |
| Locomotion | `go2-flat-terrain-walk` | Unitree Go2 flat-terrain walking |
| Locomotion | `anymal_c_locomotion_flat` | ANYmal-C flat-terrain locomotion |
| Navigation | `anymal-c-flat-terrain-nav`, `anymal_c_navigation_flat` | ANYmal-C flat-terrain position and heading navigation |
| Navigation | `anymal_c_navigation_rough` | ANYmal-C rough-terrain position and heading navigation |
| Manipulation | `franka_lift_cube` | Franka cube lifting |
| Manipulation | `franka_open_cabinet` | Franka cabinet opening |

## 🚀 Quick Start

> The following examples use the Python project management tool: [UV](https://docs.astral.sh/uv/)
>
> Before starting, please [install](https://docs.astral.sh/uv/getting-started/installation/) this tool.

### Clone Repository

```bash
git clone https://github.com/Motphys/MotrixLab

cd MotrixLab

git lfs pull
```

### Install Dependencies

Install all dependencies:

```bash
uv sync --all-packages --all-extras
```

SKRL framework supports JAX(Flax) or PyTorch as training backends. You can also choose to install only one training backend based on your hardware environment:

Install JAX as training backend (Linux only):

```bash
uv sync --all-packages --extra skrl-jax
```

Install PyTorch as training backend:

```bash
uv sync --all-packages --extra skrl-torch
```

## 🎯 Usage Guide

### Environment Visualization

View environments without executing training:

```bash
uv run scripts/view.py --env cartpole
```

View the ANYmal-C rough-terrain navigation task:

```bash
uv run scripts/view.py --env anymal_c_navigation_rough --num-envs 1
```

### Model Training

```bash
uv run scripts/train.py --env cartpole
```

Training results are saved in the `runs/{env-name}/` directory.

View training data through TensorBoard:

```bash
uv run tensorboard --logdir runs/{env-name}
```

Train the ANYmal-C rough-terrain navigation task:

```bash
uv run scripts/train.py --env anymal_c_navigation_rough --num-envs 2048 --train-backend jax
```

Continue training from a JAX PPO checkpoint:

```bash
uv run scripts/train.py --env anymal_c_navigation_rough --pretrained path/to/checkpoint.pickle
```

### Model Inference

```bash
uv run scripts/play.py --env cartpole
```

To load a specific trained policy:

```bash
uv run scripts/play.py --env anymal_c_navigation_rough --policy path/to/best_agent.pickle --num-envs 64
```

## ANYmal-C Rough-terrain Navigation

The `anymal_c_navigation_rough` environment migrates the Isaac Lab `isaac-navigation-flat-anymal-c-v0` task into MotrixLab and replaces the flat plane with a complex heightfield terrain.

![ANYmal-C rough-terrain navigation demo](docs/source/_static/images/demo/anymal_c_rough_navigation.gif)

-   **Action space**: `Box(-1.0, 1.0, (12,), float32)` for the 12 ANYmal-C joint position commands
-   **Observation space**: `Box(-inf, inf, (54,), float32)`
-   **Terrain**: 40 m x 40 m MuJoCo heightfield from `heightmap.png`, with world height from -1.0 m to 1.5 m
-   **Task**: Navigate to a sampled XY target and target yaw, then stop stably on the terrain
-   **Episode length**: 1500 control steps, or 15 seconds at the default 100 Hz control rate
-   **Reset and targets**: Samples spawn and goal positions with terrain-height lookup, height-difference limits, slope checks, and safe fallback targets
-   **Rewards**: Extends the flat navigation reward with vertical-velocity stability, body-orientation stability, and foot-contact rewards
-   **Termination**: Includes physics failure checks and a 19 m terrain-boundary check

For more usage methods, please refer to the [User Documentation](https://motrixlab.readthedocs.io)

## 📬 Contact

Have questions or suggestions? Feel free to contact us through:

-   GitHub Issues: [Submit Issues](https://github.com/Motphys/MotrixLab/issues)
-   Discussions: [Join Discussion](https://github.com/Motphys/MotrixLab/discussions)
