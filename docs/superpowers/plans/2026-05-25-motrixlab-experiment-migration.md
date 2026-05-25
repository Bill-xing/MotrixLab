# MotrixLab Experiment Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the experimental MotrixLab environments and transfer-learning support from `<source-motrixlab-repo>` into the clean Git repository at `<target-motrixlab-repo>` while keeping the result suitable for open-source release.

**Architecture:** The target Git repository remains the base. Environment code and assets are merged into `motrix_envs`, RL configuration and trainer changes are merged into `motrix_rl`, and task wrappers are merged into `train_eval_scripts`. Local run outputs, logs, pyc caches, and private experiment artifacts are excluded.

**Tech Stack:** Python 3.10, MotrixSim NumPy backend, SKRL PPO, JAX/Torch trainer entrypoints, absl flags, shell wrappers, py_compile, pytest-style tests when pytest is available.

---

## File Map

**Create**
- `motrix_envs/tests/test_migrated_registry_contracts.py`: contract tests for migrated environment names, RL config registration, and transfer-learning source hooks.
- `motrix_envs/src/motrix_envs/locomotion/anymal_c/`: ANYmal-C locomotion flat task and XML assets.
- `motrix_envs/src/motrix_envs/locomotion/go2/`: Go2 flat walking task and XML assets.
- `motrix_envs/src/motrix_envs/manipulation/`: Franka lift cube and open cabinet tasks with assets.
- `motrix_envs/src/motrix_envs/math/`: shared quaternion utilities.
- `train_eval_scripts/`: task-specific train/eval shell wrappers.

**Modify**
- `motrix_envs/src/motrix_envs/navigation/anymal_c/`: migrate flat and rough navigation implementation while preserving target-only helper/test files where still useful.
- `motrix_envs/src/motrix_envs/locomotion/go1/`: merge rough terrain and stairs terrain support.
- `motrix_envs/src/motrix_envs/__init__.py`: import `manipulation` as a registered package.
- `motrix_envs/src/motrix_envs/locomotion/__init__.py`: import `anymal_c`, `go1`, and `go2`.
- `motrix_envs/src/motrix_envs/navigation/__init__.py`: keep ANYmal-C navigation import.
- `motrix_envs/pyproject.toml`: add `scipy>=1.15.3`.
- `motrix_rl/src/motrix_rl/cfgs.py`: add migrated RL configs and preserve legacy ANYmal-C alias.
- `motrix_rl/src/motrix_rl/skrl/jax/train/ppo.py`: add optional pretrained checkpoint loading.
- `scripts/train.py`: add `--pretrained` and pass it to JAX trainer.
- `uv.lock`: update from source lock or regenerate if package metadata requires it.
- `.gitignore`: extend only if a migrated generated artifact is not already ignored.

**Preserve**
- `.gitignore`, `README.md`, `README.zh-CN.md`, `CLAUDE.md`, root release metadata.
- `scripts/run_isaaclab_anymalc.sh` and `scripts/run_motrixlab_go1.sh`.
- `motrix_envs/src/motrix_envs/navigation/anymal_c/reward_utils.py` unless static checks show the helper is unused and obsolete.
- `motrix_envs/tests/test_anymal_c_rewards.py` as a narrow regression test for termination-penalty helper behavior.

---

### Task 1: Add Migration Contract Tests

**Files:**
- Create: `motrix_envs/tests/test_migrated_registry_contracts.py`

- [ ] **Step 1: Write the failing test**

Create `motrix_envs/tests/test_migrated_registry_contracts.py` with this content:

```python
from pathlib import Path

import motrix_envs  # noqa: F401
from motrix_envs import registry as env_registry
from motrix_rl import cfgs as rl_cfgs  # noqa: F401
from motrix_rl import registry as rl_registry


EXPECTED_ENV_BACKENDS = {
    "anymal-c-flat-terrain-nav": ["np"],
    "anymal_c_navigation_flat": ["np"],
    "anymal_c_navigation_rough": ["np"],
    "anymal_c_locomotion_flat": ["np"],
    "go1-flat-terrain-walk": ["np"],
    "go1-rough-terrain-walk": ["np"],
    "go1-stairs-terrain-walk": ["np"],
    "go2-flat-terrain-walk": ["np"],
    "franka_lift_cube": ["np"],
    "franka_open_cabinet": ["np"],
}


EXPECTED_RL_CONFIGS = {
    "anymal-c-flat-terrain-nav",
    "anymal_c_navigation_flat",
    "anymal_c_navigation_rough",
    "go1-flat-terrain-walk",
    "go1-rough-terrain-walk",
    "go1-stairs-terrain-walk",
    "go2-flat-terrain-walk",
    "franka_lift_cube",
    "franka_open_cabinet",
}


def test_migrated_environments_are_registered_with_np_backend():
    registered = env_registry.list_registered_envs()

    for env_name, expected_backends in EXPECTED_ENV_BACKENDS.items():
        assert env_name in registered
        assert registered[env_name]["available_backends"] == expected_backends


def test_migrated_environments_have_jax_rl_configs():
    for env_name in EXPECTED_RL_CONFIGS:
        cfg = rl_registry.default_rl_cfg(env_name, "skrl", "jax")

        assert cfg.num_envs > 0
        assert cfg.rollouts > 0
        assert cfg.max_batch_env_steps > 0


def test_train_script_exposes_pretrained_flag():
    source = Path("scripts/train.py").read_text(encoding="utf-8")

    assert 'flags.DEFINE_string("pretrained"' in source
    assert "trainer.train(pretrained=pretrained_path)" in source


def test_jax_trainer_loads_pretrained_checkpoint_before_training():
    source = Path("motrix_rl/src/motrix_rl/skrl/jax/train/ppo.py").read_text(encoding="utf-8")

    assert "def train(self, pretrained: str | None = None)" in source
    assert "agent.load(pretrained)" in source
    assert "SequentialTrainer" in source
```

- [ ] **Step 2: Run the test to verify it fails before migration**

Run:

```bash
python -m pytest motrix_envs/tests/test_migrated_registry_contracts.py -q
```

Expected: fail because migrated environments such as `anymal_c_navigation_rough`, `go2-flat-terrain-walk`, and `franka_lift_cube` are not registered yet. If pytest is not installed, run this fallback and record that pytest is unavailable:

```bash
python - <<'PY'
from pathlib import Path
path = Path("motrix_envs/tests/test_migrated_registry_contracts.py")
compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("contract test syntax ok")
PY
```

- [ ] **Step 3: Commit the failing contract test**

Run:

```bash
git add motrix_envs/tests/test_migrated_registry_contracts.py
git commit -m "test: define migrated environment contracts"
```

---

### Task 2: Migrate Extended Locomotion, Manipulation, and Math Packages

**Files:**
- Create: `motrix_envs/src/motrix_envs/locomotion/anymal_c/`
- Create: `motrix_envs/src/motrix_envs/locomotion/go2/`
- Create: `motrix_envs/src/motrix_envs/manipulation/`
- Create: `motrix_envs/src/motrix_envs/math/`
- Modify: `motrix_envs/src/motrix_envs/locomotion/go1/`
- Modify: `motrix_envs/src/motrix_envs/__init__.py`
- Modify: `motrix_envs/src/motrix_envs/locomotion/__init__.py`

- [ ] **Step 1: Copy only functional source directories**

Run:

```bash
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
  <source-motrixlab-repo>/motrix_envs/src/motrix_envs/locomotion/anymal_c \
  motrix_envs/src/motrix_envs/locomotion/

rsync -a --exclude='__pycache__' --exclude='*.pyc' \
  <source-motrixlab-repo>/motrix_envs/src/motrix_envs/locomotion/go2 \
  motrix_envs/src/motrix_envs/locomotion/

rsync -a --exclude='__pycache__' --exclude='*.pyc' \
  <source-motrixlab-repo>/motrix_envs/src/motrix_envs/manipulation \
  motrix_envs/src/motrix_envs/

rsync -a --exclude='__pycache__' --exclude='*.pyc' \
  <source-motrixlab-repo>/motrix_envs/src/motrix_envs/math \
  motrix_envs/src/motrix_envs/

rsync -a --exclude='__pycache__' --exclude='*.pyc' \
  <source-motrixlab-repo>/motrix_envs/src/motrix_envs/locomotion/go1/common.py \
  <source-motrixlab-repo>/motrix_envs/src/motrix_envs/locomotion/go1/walk_rough_terrain.py \
  <source-motrixlab-repo>/motrix_envs/src/motrix_envs/locomotion/go1/walk_stairs_terrain.py \
  motrix_envs/src/motrix_envs/locomotion/go1/

rsync -a --exclude='__pycache__' --exclude='*.pyc' \
  <source-motrixlab-repo>/motrix_envs/src/motrix_envs/locomotion/go1/xmls/ \
  motrix_envs/src/motrix_envs/locomotion/go1/xmls/
```

This copies the rough/stairs scenes, heightmap assets, `materials.xml`, and the source version of the Go1 XML mesh assets used by those scenes.

- [ ] **Step 2: Merge Go1 config and implementation changes**

Apply the source changes from these files, keeping existing target formatting and comments concise:

```text
<source-motrixlab-repo>/motrix_envs/src/motrix_envs/locomotion/go1/cfg.py
<source-motrixlab-repo>/motrix_envs/src/motrix_envs/locomotion/go1/walk_np.py
```

The resulting target files must include:

```python
@registry.envcfg("go1-rough-terrain-walk")
@dataclass
class Go1WalkNpRoughEnvCfg(Go1WalkNpEnvCfg):
    render_spacing: float = 0.0
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_rough_terrain.xml"


@registry.envcfg("go1-stairs-terrain-walk")
@dataclass
class Go1WalkNpStairsEnvCfg(Go1WalkNpEnvCfg):
    render_spacing: float = 0.0
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs_terrain.xml"
```

The resulting `walk_np.py` must use:

```python
from motrix_envs.math.quaternion import Quaternion
```

and replace local gravity rotation calls with:

```python
Quaternion.rotate_inverse(base_quat, self.gravity_vec)
```

- [ ] **Step 3: Update package imports**

Set `motrix_envs/src/motrix_envs/__init__.py` to import manipulation:

```python
from . import basic, locomotion, manipulation, navigation  # noqa: F401
```

Set `motrix_envs/src/motrix_envs/locomotion/__init__.py` to import migrated locomotion packages:

```python
from . import anymal_c, go1, go2  # noqa: F401 register envs
```

- [ ] **Step 4: Normalize migrated text files to LF**

Run:

```bash
python - <<'PY'
from pathlib import Path

roots = [
    Path("motrix_envs/src/motrix_envs/locomotion/anymal_c"),
    Path("motrix_envs/src/motrix_envs/locomotion/go1"),
    Path("motrix_envs/src/motrix_envs/locomotion/go2"),
    Path("motrix_envs/src/motrix_envs/manipulation"),
    Path("motrix_envs/src/motrix_envs/math"),
]
text_suffixes = {".py", ".xml", ".md", ".toml", ".yaml", ".yml", ".urdf", ".mtl", ".obj", ".hfield"}
for root in roots:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in text_suffixes:
            data = path.read_bytes()
            path.write_bytes(data.replace(b"\r\n", b"\n"))
PY
```

- [ ] **Step 5: Verify syntax for migrated packages**

Run:

```bash
python -m py_compile \
  motrix_envs/src/motrix_envs/math/quaternion.py \
  motrix_envs/src/motrix_envs/locomotion/anymal_c/cfg.py \
  motrix_envs/src/motrix_envs/locomotion/anymal_c/anymal_c_np.py \
  motrix_envs/src/motrix_envs/locomotion/go1/cfg.py \
  motrix_envs/src/motrix_envs/locomotion/go1/walk_np.py \
  motrix_envs/src/motrix_envs/locomotion/go1/walk_rough_terrain.py \
  motrix_envs/src/motrix_envs/locomotion/go1/walk_stairs_terrain.py \
  motrix_envs/src/motrix_envs/locomotion/go2/cfg.py \
  motrix_envs/src/motrix_envs/locomotion/go2/walk_np.py \
  motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/cfg.py \
  motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/franka_lift_cube_np.py \
  motrix_envs/src/motrix_envs/manipulation/franka_open_cabinet/cfg.py \
  motrix_envs/src/motrix_envs/manipulation/franka_open_cabinet/franka_open_cabinet_np.py
```

Expected: exit code 0.

- [ ] **Step 6: Commit extended environment migration**

Run:

```bash
git add motrix_envs/src/motrix_envs
git commit -m "feat: add extended locomotion and manipulation environments"
```

---

### Task 3: Migrate ANYmal-C Navigation Flat and Rough Tasks

**Files:**
- Modify: `motrix_envs/src/motrix_envs/navigation/anymal_c/`
- Modify: `motrix_rl/src/motrix_rl/cfgs.py`
- Preserve: `motrix_envs/src/motrix_envs/navigation/anymal_c/reward_utils.py`
- Preserve or update: `motrix_envs/tests/test_anymal_c_rewards.py`

- [ ] **Step 1: Copy ANYmal-C navigation code and assets without generated caches**

Run:

```bash
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
  <source-motrixlab-repo>/motrix_envs/src/motrix_envs/navigation/anymal_c/ \
  motrix_envs/src/motrix_envs/navigation/anymal_c/
```

Confirm the target-only helper still exists:

```bash
test -f motrix_envs/src/motrix_envs/navigation/anymal_c/reward_utils.py
test -f motrix_envs/tests/test_anymal_c_rewards.py
```

- [ ] **Step 2: Add legacy environment alias for open-source compatibility**

In `motrix_envs/src/motrix_envs/navigation/anymal_c/cfg.py`, the flat config must be registered under both the migrated source name and the target repository legacy name:

The decorator stack immediately before `class AnymalCEnvCfg(EnvCfg):` must be:

```python
@registry.envcfg("anymal-c-flat-terrain-nav")
@registry.envcfg("anymal_c_navigation_flat")
@dataclass
```

In `motrix_envs/src/motrix_envs/navigation/anymal_c/anymal_c_np.py`, the flat environment class must be registered under both names:

The decorator stack immediately before `class AnymalCEnv(NpEnv):` must be:

```python
@registry.env("anymal-c-flat-terrain-nav", "np")
@registry.env("anymal_c_navigation_flat", "np")
```

Keep the rough terrain registration:

The decorator immediately before `class AnymalCRoughEnv(AnymalCEnv):` must remain:

```python
@registry.env("anymal_c_navigation_rough", "np")
```

- [ ] **Step 3: Clean comments and public-facing prose**

Run this scan:

```bash
rg -n "每个函数|说明:|参数与关键变量|执行流程|修改实验|周报|路径替换|请替换|待补|占位" \
  motrix_envs/src/motrix_envs/navigation/anymal_c \
  motrix_envs/src/motrix_envs/locomotion \
  motrix_envs/src/motrix_envs/manipulation \
  motrix_rl/src/motrix_rl/cfgs.py \
  scripts/train.py || true
```

For every match in migrated code, either delete the note or rewrite it as a concise implementation comment. Keep comments that explain terrain sampling, reward shaping, quaternion math, or non-obvious simulator constraints.

- [ ] **Step 4: Normalize ANYmal-C navigation text files to LF**

Run:

```bash
python - <<'PY'
from pathlib import Path

root = Path("motrix_envs/src/motrix_envs/navigation/anymal_c")
text_suffixes = {".py", ".xml", ".md", ".toml", ".yaml", ".yml", ".obj", ".mtl"}
for path in root.rglob("*"):
    if path.is_file() and path.suffix in text_suffixes:
        data = path.read_bytes()
        path.write_bytes(data.replace(b"\r\n", b"\n"))
PY
```

- [ ] **Step 5: Merge ANYmal-C RL configs and preserve legacy alias**

In `motrix_rl/src/motrix_rl/cfgs.py`, the navigation config class for flat ANYmal-C must register both names:

```python
class navigation:
    @rlcfg("anymal-c-flat-terrain-nav")
    @rlcfg("anymal_c_navigation_flat")
    @dataclass
    class AnymalCPPOConfig(PPOCfg):
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 100_000_000
        check_point_interval: int = 100
        learning_rate: float = 3e-4
        rollouts: int = 48
        learning_epochs: int = 6
        mini_batches: int = 32
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0
        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True
        learning_rate_scheduler_kl_threshold: float = 0.02
        entropy_loss_scale: float = 0.005
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("anymal_c_navigation_rough")
    @dataclass
    class AnymalCRoughPPOConfig(AnymalCPPOConfig):
        max_env_steps: int = 200_000_000
        learning_rate: float = 3e-4
        rollouts: int = 48
        learning_epochs: int = 6
        mini_batches: int = 32
        learning_rate_scheduler_kl_threshold: float = 0.02
        entropy_loss_scale: float = 0.03
```

- [ ] **Step 6: Verify migrated ANYmal-C syntax**

Run:

```bash
python -m py_compile \
  motrix_envs/src/motrix_envs/navigation/anymal_c/cfg.py \
  motrix_envs/src/motrix_envs/navigation/anymal_c/anymal_c_np.py \
  motrix_envs/src/motrix_envs/navigation/anymal_c/reward_utils.py \
  motrix_rl/src/motrix_rl/cfgs.py
```

Expected: exit code 0.

- [ ] **Step 7: Commit ANYmal-C migration**

Run:

```bash
git add motrix_envs/src/motrix_envs/navigation/anymal_c motrix_envs/tests/test_anymal_c_rewards.py motrix_rl/src/motrix_rl/cfgs.py
git commit -m "feat: migrate ANYmal-C flat and rough navigation tasks"
```

---

### Task 4: Add Transfer-Learning Checkpoint Loading

**Files:**
- Modify: `scripts/train.py`
- Modify: `motrix_rl/src/motrix_rl/skrl/jax/train/ppo.py`

- [ ] **Step 1: Update train script flag parsing**

In `scripts/train.py`, add this flag near the other training flags:

```python
_PRETRAINED = flags.DEFINE_string("pretrained", None, "Path to a pretrained checkpoint for transfer learning")
```

After trainer creation and before training, pass the value to JAX trainer:

```python
pretrained_path = _PRETRAINED.value
if pretrained_path:
    logger.info("Loading pretrained checkpoint: %s", pretrained_path)

trainer.train(pretrained=pretrained_path)
```

Replace the existing final call:

```python
trainer.train()
```

with the `trainer.train(pretrained=pretrained_path)` call above.

- [ ] **Step 2: Update JAX trainer signature and loading behavior**

In `motrix_rl/src/motrix_rl/skrl/jax/train/ppo.py`, change:

```python
def train(self) -> None:
```

to:

```python
def train(self, pretrained: str | None = None) -> None:
```

After creating the agent and before constructing `SequentialTrainer`, add:

```python
if pretrained:
    print(f"[Transfer Learning] Loading pretrained weights from: {pretrained}")
    agent.load(pretrained)
    print("[Transfer Learning] Pretrained weights loaded successfully")
```

- [ ] **Step 3: Keep Torch path behavior explicit**

Torch trainer calls from `scripts/train.py` receive the same `pretrained` argument. If the Torch trainer does not support the argument, branch the final call by backend:

```python
if train_backend == "jax":
    trainer.train(pretrained=pretrained_path)
else:
    if pretrained_path:
        raise ValueError("--pretrained is currently supported only with --train-backend=jax")
    trainer.train()
```

This preserves current Torch behavior and gives users a clear error instead of a Python signature crash.

- [ ] **Step 4: Verify source contracts**

Run:

```bash
python - <<'PY'
from pathlib import Path

train_source = Path("scripts/train.py").read_text(encoding="utf-8")
ppo_source = Path("motrix_rl/src/motrix_rl/skrl/jax/train/ppo.py").read_text(encoding="utf-8")

assert 'flags.DEFINE_string("pretrained"' in train_source
assert "trainer.train(pretrained=pretrained_path)" in train_source
assert "def train(self, pretrained: str | None = None)" in ppo_source
assert "agent.load(pretrained)" in ppo_source
print("transfer learning source contracts ok")
PY
```

Expected: prints `transfer learning source contracts ok`.

- [ ] **Step 5: Commit transfer-learning support**

Run:

```bash
git add scripts/train.py motrix_rl/src/motrix_rl/skrl/jax/train/ppo.py
git commit -m "feat: add transfer-learning checkpoint loading"
```

---

### Task 5: Merge Project Metadata and Task Wrapper Scripts

**Files:**
- Modify: `motrix_envs/pyproject.toml`
- Modify: `uv.lock`
- Create: `train_eval_scripts/`
- Modify: `.gitignore` only if generated artifacts are not already ignored

- [ ] **Step 1: Add scipy dependency**

In `motrix_envs/pyproject.toml`, ensure the dependencies list contains:

```toml
dependencies = [
    "motrixsim>=0.4.0",
    "scipy>=1.15.3",
]
```

- [ ] **Step 2: Copy task wrapper scripts**

Run:

```bash
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
  <source-motrixlab-repo>/train_eval_scripts/ \
  train_eval_scripts/
```

- [ ] **Step 3: Normalize shell scripts and make them executable**

Run:

```bash
python - <<'PY'
from pathlib import Path

for path in Path("train_eval_scripts").rglob("*.bash"):
    data = path.read_bytes()
    path.write_bytes(data.replace(b"\r\n", b"\n"))
PY

chmod +x train_eval_scripts/*/*.bash
```

- [ ] **Step 4: Keep root project version from target**

Do not copy the source root `pyproject.toml` version downgrade. The target must keep:

```toml
version = "0.0.2"
```

- [ ] **Step 5: Update or verify lock file**

If `uv` is available, run:

```bash
uv lock
```

If `uv` is not available, copy the source `uv.lock` only after confirming it includes `scipy` and does not downgrade the root package metadata:

```bash
rg -n 'name = "scipy"|version = "0.0.1"|version = "0.0.2"' <source-motrixlab-repo>/uv.lock
```

Expected: `scipy` is present. Do not accept a root package downgrade.

- [ ] **Step 6: Verify shell wrappers**

Run:

```bash
bash -n train_eval_scripts/anymal_c_navigation_flat/train.bash
bash -n train_eval_scripts/anymal_c_navigation_flat/eval.bash
bash -n train_eval_scripts/franka_lift_cube/train.bash
bash -n train_eval_scripts/franka_lift_cube/eval.bash
bash -n train_eval_scripts/franka_open_cabinet/train.bash
bash -n train_eval_scripts/franka_open_cabinet/eval.bash
bash -n scripts/run_isaaclab_anymalc.sh
bash -n scripts/run_motrixlab_go1.sh
```

Expected: exit code 0 for every script.

- [ ] **Step 7: Commit metadata and scripts**

Run:

```bash
git add motrix_envs/pyproject.toml uv.lock train_eval_scripts .gitignore
git commit -m "chore: normalize migrated project metadata"
```

---

### Task 6: Final Cleanup and Verification

**Files:**
- Modify: any files flagged by verification.
- Test: `motrix_envs/tests/test_migrated_registry_contracts.py`
- Test: `motrix_envs/tests/test_anymal_c_rewards.py`

- [ ] **Step 1: Remove generated caches and local artifacts from the worktree**

Run:

```bash
find motrix_envs motrix_rl scripts train_eval_scripts -type d -name '__pycache__' -prune -exec rm -rf {} +
find motrix_envs motrix_rl scripts train_eval_scripts -type f -name '*.pyc' -delete
```

- [ ] **Step 2: Check excluded artifacts are not staged or tracked**

Run:

```bash
git status --short --untracked-files=all
git ls-files | rg '(^|/)(__pycache__|.*\.pyc$|runs/|training_log\.txt$|^image/|magic\.mgc$|\.venv/)' && exit 1 || true
```

Expected: no excluded artifact paths are listed by `git ls-files`.

- [ ] **Step 3: Check CRLF line endings in migrated text files**

Run:

```bash
python - <<'PY'
from pathlib import Path

roots = [
    Path("motrix_envs/src/motrix_envs"),
    Path("motrix_rl/src/motrix_rl"),
    Path("scripts"),
    Path("train_eval_scripts"),
]
text_suffixes = {".py", ".xml", ".md", ".toml", ".yaml", ".yml", ".urdf", ".mtl", ".obj", ".bash", ".sh"}
bad = []
for root in roots:
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in text_suffixes and b"\r\n" in path.read_bytes():
            bad.append(str(path))
if bad:
    raise SystemExit("CRLF line endings found:\n" + "\n".join(bad))
print("line endings ok")
PY
```

Expected: prints `line endings ok`.

- [ ] **Step 4: Run syntax verification**

Run:

```bash
python -m py_compile \
  motrix_envs/src/motrix_envs/__init__.py \
  motrix_envs/src/motrix_envs/locomotion/__init__.py \
  motrix_envs/src/motrix_envs/navigation/__init__.py \
  motrix_envs/src/motrix_envs/navigation/anymal_c/cfg.py \
  motrix_envs/src/motrix_envs/navigation/anymal_c/anymal_c_np.py \
  motrix_envs/src/motrix_envs/locomotion/anymal_c/cfg.py \
  motrix_envs/src/motrix_envs/locomotion/anymal_c/anymal_c_np.py \
  motrix_envs/src/motrix_envs/locomotion/go1/cfg.py \
  motrix_envs/src/motrix_envs/locomotion/go1/walk_np.py \
  motrix_envs/src/motrix_envs/locomotion/go1/walk_rough_terrain.py \
  motrix_envs/src/motrix_envs/locomotion/go1/walk_stairs_terrain.py \
  motrix_envs/src/motrix_envs/locomotion/go2/cfg.py \
  motrix_envs/src/motrix_envs/locomotion/go2/walk_np.py \
  motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/cfg.py \
  motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/franka_lift_cube_np.py \
  motrix_envs/src/motrix_envs/manipulation/franka_open_cabinet/cfg.py \
  motrix_envs/src/motrix_envs/manipulation/franka_open_cabinet/franka_open_cabinet_np.py \
  motrix_envs/src/motrix_envs/math/quaternion.py \
  motrix_rl/src/motrix_rl/cfgs.py \
  motrix_rl/src/motrix_rl/skrl/jax/train/ppo.py \
  scripts/train.py
```

Expected: exit code 0.

- [ ] **Step 5: Run contract tests**

Run:

```bash
python -m pytest motrix_envs/tests/test_anymal_c_rewards.py motrix_envs/tests/test_migrated_registry_contracts.py -q
```

Expected: pass. If pytest is unavailable, run:

```bash
python - <<'PY'
import numpy as np
import motrix_envs
from motrix_envs import registry as env_registry
from motrix_envs.navigation.anymal_c.reward_utils import apply_termination_penalty
from motrix_rl import cfgs as rl_cfgs
from motrix_rl import registry as rl_registry

expected_envs = [
    "anymal-c-flat-terrain-nav",
    "anymal_c_navigation_flat",
    "anymal_c_navigation_rough",
    "anymal_c_locomotion_flat",
    "go1-flat-terrain-walk",
    "go1-rough-terrain-walk",
    "go1-stairs-terrain-walk",
    "go2-flat-terrain-walk",
    "franka_lift_cube",
    "franka_open_cabinet",
]
registered = env_registry.list_registered_envs()
for name in expected_envs:
    assert registered[name]["available_backends"] == ["np"], name
    assert rl_registry.default_rl_cfg(name, "skrl", "jax").num_envs > 0, name

actual = apply_termination_penalty(
    np.array([10.0, 5.0], dtype=np.float32),
    np.array([True, False]),
    -200.0,
)
np.testing.assert_allclose(actual, np.array([-190.0, 5.0], dtype=np.float32))
print("fallback migration verification ok")
PY
```

Expected: prints `fallback migration verification ok`.

- [ ] **Step 6: Run Git whitespace and artifact checks**

Run:

```bash
git diff --check
git status --short --untracked-files=all
```

Expected: `git diff --check` exits 0. `git status` lists only intentional source, asset, script, test, metadata, and doc files.

- [ ] **Step 7: Commit final tests and cleanup**

Run:

```bash
git add motrix_envs/tests motrix_envs/src motrix_rl/src scripts train_eval_scripts motrix_envs/pyproject.toml uv.lock .gitignore
git commit -m "test: cover migrated environment registrations"
```

If there are no remaining changes after earlier commits, record that no final cleanup commit is needed.

---

### Task 7: Public-Release Readiness Check

**Files:**
- No required edits unless checks reveal a release blocker.

- [ ] **Step 1: Inspect commit history**

Run:

```bash
git log --oneline --decorate -10
```

Expected: recent commits are topic-based and public-reviewable:

```text
test: cover migrated environment registrations
chore: normalize migrated project metadata
feat: add transfer-learning checkpoint loading
feat: migrate ANYmal-C flat and rough navigation tasks
feat: add extended locomotion and manipulation environments
test: define migrated environment contracts
docs: add MotrixLab experiment migration design
```

- [ ] **Step 2: Verify no excluded source artifacts are tracked**

Run:

```bash
git ls-files | rg '(^|/)(__pycache__|.*\.pyc$|runs/|training_log\.txt$|^image/|magic\.mgc$|\.venv/)' && exit 1 || true
```

Expected: no output.

- [ ] **Step 3: Record verification limitations**

If full simulator training is not run, record this in the final response:

```text
Full MotrixSim training was not executed because it requires simulator/runtime/GPU setup and is outside this migration verification scope.
```

- [ ] **Step 4: Stop before pushing**

Do not push to GitHub unless the user explicitly asks for the migrated version to be uploaded. Report local commit hashes and verification results first.
