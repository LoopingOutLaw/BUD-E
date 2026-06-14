"""Tests for the MJX UR5e arm environment wrapper."""
import numpy as np
from bude_vla.envs.so101_mjx import UR5eMJMJX


def test_env_loads():
    env = UR5eMJMJX()
    assert env is not None
    # 6 arm joints + 7 qpos for 1 slide joint = 7 actuator-controlled
    assert env.model_mj.nu >= 7


def test_action_dim_matches_spec():
    env = UR5eMJMJX()
    assert env.action_dim == env.model_mj.nu


def test_reset_returns_mjx_data_with_correct_dim():
    env = UR5eMJMJX()
    s = env.reset()
    nq = int(env.model.nq)
    assert int(s.qpos.shape[0]) == nq


def test_step_changes_state():
    env = UR5eMJMJX()
    s = env.reset()
    s_before = np.asarray(s.qpos)
    action = np.zeros(env.model_mj.nu)
    # Move shoulder to draw an arm
    action[0] = 1.0
    s_new = env.step_static(s, action)
    assert not np.allclose(np.asarray(s_new.qpos), s_before, atol=1e-4)


def test_action_bounds_have_correct_shape():
    env = UR5eMJMJX()
    lo, hi = env.action_bounds()
    assert lo.shape[0] == env.model_mj.nu
    assert hi.shape[0] == env.model_mj.nu
    assert (hi > lo).all()


def test_ender_returns_correct_shape():
    env = UR5eMJMJX()
    s = env.reset()
    img = env.render(s, height=64, width=64)
    assert img.shape == (64, 64, 3)
    # Image should be valid (no NaN)
    assert not np.isnan(img).any()
