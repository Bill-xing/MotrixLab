from pathlib import Path

import motrix_envs  # noqa: F401
from motrix_envs import registry as env_registry


REPO_ROOT = Path(__file__).resolve().parents[2]

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

ENVS_WITHOUT_JAX_RL_CONFIG = {"anymal_c_locomotion_flat"}
EXPECTED_RL_CONFIGS = set(EXPECTED_ENV_BACKENDS) - ENVS_WITHOUT_JAX_RL_CONFIG


def test_migrated_environments_are_registered_with_np_backend():
    registered = env_registry.list_registered_envs()

    for env_name, expected_backends in EXPECTED_ENV_BACKENDS.items():
        assert env_name in registered
        assert registered[env_name]["available_backends"] == expected_backends


def test_migrated_environments_have_jax_rl_configs():
    from motrix_rl import cfgs as rl_cfgs  # noqa: F401
    from motrix_rl import registry as rl_registry

    for env_name in EXPECTED_RL_CONFIGS:
        cfg = rl_registry.default_rl_cfg(env_name, "skrl", "jax")

        assert cfg.num_envs > 0
        assert cfg.rollouts > 0
        assert cfg.max_batch_env_steps > 0


def test_train_script_exposes_pretrained_flag():
    source = (REPO_ROOT / "scripts/train.py").read_text(encoding="utf-8")

    assert 'flags.DEFINE_string("pretrained"' in source
    assert "trainer.train(pretrained=pretrained_path)" in source


def test_jax_trainer_loads_pretrained_checkpoint_before_training():
    source = (REPO_ROOT / "motrix_rl/src/motrix_rl/skrl/jax/train/ppo.py").read_text(encoding="utf-8")

    assert "def train(self, pretrained: str | None = None)" in source
    assert "agent.load(pretrained)" in source
    assert "SequentialTrainer" in source
