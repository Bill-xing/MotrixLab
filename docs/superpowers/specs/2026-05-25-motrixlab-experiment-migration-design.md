# MotrixLab Experiment Migration Design

Date: 2026-05-25

## Goal

Migrate the useful experimental MotrixLab work from `/home/xing/崎岖地形导航迁移任务/MotrixLab` into the clean Git repository at `/home/xing/isaaclab_motrixlab/MotrixLab`, then prepare the result as an open-source-ready version.

The migration should preserve the current repository's open-source cleanup work and add the experimental environments, training configuration, and transfer-learning support from the source directory. It should not copy local training outputs or private/temporary artifacts.

## Current Context

The target repository is a Git repository on `main`, tracking `github/main`. Its current public history contains a clean open-source release and a basic ANYmal-C navigation adaptation.

The source directory is not a Git repository. It contains more experimental functionality than the target repository, including rough-terrain navigation, additional locomotion environments, manipulation tasks, training scripts, checkpoint loading support, and local run artifacts.

The source directory also contains CRLF line endings in some files and local experimental outputs. Those should be normalized or excluded before commit.

## Migration Scope

Migrate these functional areas:

1. ANYmal-C navigation flat and rough terrain tasks.
2. ANYmal-C locomotion flat task.
3. Go1 rough-terrain and stairs-terrain walking tasks.
4. Go2 flat-terrain walking task.
5. Franka manipulation tasks:
   - `franka_lift_cube`
   - `franka_open_cabinet`
6. Shared quaternion math utilities under `motrix_envs.math`.
7. Task helper scripts under `train_eval_scripts`.
8. `scipy` dependency for migrated terrain/math functionality.
9. JAX PPO pretrained checkpoint loading support.
10. `scripts/train.py` support for a `--pretrained` checkpoint path.
11. RL config registrations for all migrated environments.

## Excluded Content

Do not migrate these source-directory contents:

1. `runs/`
2. `training_log.txt`
3. `image/`
4. `magic.mgc`
5. `.git/`
6. `.venv/`
7. `__pycache__/`
8. `*.pyc`
9. Temporary logs, generated training outputs, and local-only experiment artifacts.

Do not overwrite the target repository's existing open-source metadata unless there is a concrete reason:

1. `.gitignore`
2. `README.md`
3. `README.zh-CN.md`
4. `CLAUDE.md`
5. Existing release runner scripts
6. Existing tests and helper utilities

## Open-Source Cleanup Requirements

Normalize migrated text files to LF line endings.

Keep comments concise. Preserve comments that explain non-obvious simulation, terrain, reward, or quaternion logic. Remove or rewrite comments that are experiment-note style, report-like, or merely restate each line of code.

Keep public files in English where they are user-facing docs or comments intended for broad open-source readers. Chinese may remain only where it is already part of established repository documentation or where it is clearly useful and not internal-progress prose.

Keep local artifacts out of Git using `.gitignore` and explicit `git status` checks.

Do not replace the target repository wholesale. The target repository remains the base; source functionality is merged into it.

## Architecture

The target repository keeps the same package structure:

- `motrix_envs` owns environment definitions, XML/MJCF assets, registry imports, and environment-level utilities.
- `motrix_rl` owns RL config registration and trainer integrations.
- `scripts` owns generic train/play/view entrypoints.
- `train_eval_scripts` owns environment-specific command wrappers.

Environment registration should happen through existing `motrix_envs.registry` decorators and package imports. Migrated environments must be importable through the package-level `motrix_envs` import path.

Transfer-learning support should remain a small extension to the JAX PPO trainer and the generic train script. It should not change Torch behavior unless required by existing code paths.

## Key File Changes

Expected files or directories to migrate or modify:

- `motrix_envs/src/motrix_envs/navigation/anymal_c/`
- `motrix_envs/src/motrix_envs/locomotion/anymal_c/`
- `motrix_envs/src/motrix_envs/locomotion/go1/`
- `motrix_envs/src/motrix_envs/locomotion/go2/`
- `motrix_envs/src/motrix_envs/manipulation/`
- `motrix_envs/src/motrix_envs/math/`
- `motrix_envs/src/motrix_envs/__init__.py`
- `motrix_envs/src/motrix_envs/locomotion/__init__.py`
- `motrix_envs/src/motrix_envs/navigation/__init__.py`
- `motrix_envs/pyproject.toml`
- `motrix_rl/src/motrix_rl/cfgs.py`
- `motrix_rl/src/motrix_rl/skrl/jax/train/ppo.py`
- `scripts/train.py`
- `train_eval_scripts/`
- `uv.lock`

Expected files or directories to preserve from the target repository:

- `motrix_envs/src/motrix_envs/navigation/anymal_c/reward_utils.py`, unless the migrated ANYmal-C implementation makes it obsolete.
- `motrix_envs/tests/test_anymal_c_rewards.py`, updated if needed to test the migrated reward boundary.
- `scripts/run_isaaclab_anymalc.sh`
- `scripts/run_motrixlab_go1.sh`
- Root README and release metadata.

## Commit Strategy

Split implementation into focused commits:

1. `feat: add extended locomotion and manipulation environments`
2. `feat: migrate ANYmal-C flat and rough navigation tasks`
3. `feat: add transfer-learning checkpoint loading`
4. `chore: normalize migrated project metadata`
5. `test: cover migrated environment registrations`

The exact commit split can be adjusted during implementation if the diff shows a cleaner boundary, but commits must remain topic-based and suitable for public review.

## Verification

Run lightweight verification that does not require full training:

1. `git diff --check`
2. `python -m py_compile` on modified Python files or package subsets.
3. Import/registry checks for migrated environment names.
4. RL config registry checks for migrated task names.
5. `bash -n` for shell scripts under `scripts/` and `train_eval_scripts/`.
6. Check that excluded artifacts are absent from `git status --short`.
7. Check for CRLF line endings in migrated text files.

Full simulator training is out of scope for this migration because it depends on MotrixSim/JAX/GPU/runtime setup and can be expensive. If a dependency is unavailable, record the exact command and failure, then run the closest static or import-level verification possible.

## Risks

The source directory is not a Git repository, so there is no source commit history to preserve.

The source implementation includes broad experimental changes. Some may be unrelated to ANYmal-C adaptation, but the selected scope intentionally includes the larger experiment feature set requested by the user.

Large XML, OBJ, STL, PNG, HFIELD, and URDF assets may be necessary for simulation environments. They should be included only when referenced by migrated scene files.

Existing target tests may need adjustment because the migrated ANYmal-C environment changes environment names, observation dimensions, reward logic, and terrain support.

## Success Criteria

The target repository contains the selected experimental environments and transfer-learning support.

The Git diff excludes local training artifacts and temporary files.

The migrated files are suitable for public review: normalized line endings, no generated output logs, no local-only path assumptions, and no excessive experiment-note prose.

Verification commands have been run and their results are recorded before claiming completion.
