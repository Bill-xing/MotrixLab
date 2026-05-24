import numpy as np

from motrix_envs.navigation.anymal_c.reward_utils import apply_termination_penalty


def test_apply_termination_penalty_preserves_terminal_penalty():
    reward = np.array([10.0, 5.0], dtype=np.float32)
    terminated = np.array([True, False])

    actual = apply_termination_penalty(reward, terminated, -200.0)

    np.testing.assert_allclose(actual, np.array([-190.0, 5.0], dtype=np.float32))
