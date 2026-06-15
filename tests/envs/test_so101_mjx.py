"""Tests for the MJX UR5e arm environment wrapper."""
import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx
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


def test_mjx_actuator_drives_joints_with_freejoint_present():
    """Demo: the cube's freejoint does NOT silently zero out motor torque.

    After 50 JIT-compiled MJX steps with shoulder ctrl = 0.5, qpos[7]
    (shoulder_pan) must move by more than 0.01 rad and the cube qpos must
    remain finite.
    """
    import jax

    env = UR5eMJMJX()
    s = env.reset()
    shoulder_before = float(s.qpos[7])
    cube_before = np.asarray(s.qpos[0:3])
    assert not np.isnan(shoulder_before), "qpos should be finite after reset"

    action = jnp.zeros(env.model_mj.nu).at[0].set(0.5)
    jit_step = jax.jit(lambda state: env.step_static(state, action))

    s = jit_step(s).replace(ctrl=action)  # warm up compile
    for _ in range(50):
        s = jit_step(s)

    shoulder_after = float(s.qpos[7])
    cube_after = np.asarray(s.qpos[0:3])

    assert not np.isnan(shoulder_after), \
        f"qpos[7] should remain finite, got {shoulder_after}"
    assert not np.any(np.isnan(cube_after)), f"cube qpos should remain finite, got {cube_after}"
    assert abs(shoulder_after - shoulder_before) > 0.01, \
        f"shoulder should move >0.01 rad with ctrl=0.5, got {shoulder_after - shoulder_before:.4f}"


def test_mjx_vmap_batched_envs_run_on_gpu():
    """Demo: vmapping step_static across batch-shape runs on GPU and produces
    distinct trajectories under distinct actions.
    """
    env = UR5eMJMJX()
    s_single = env.reset()

    batch_size = 4
    s_batch = jax.tree.map(lambda x: jnp.repeat(x[None, ...], batch_size, axis=0), s_single)

    actions_left = jnp.tile(jnp.zeros(env.model_mj.nu).at[0].set(-0.5)[None, :], (batch_size, 1))
    actions_right = jnp.tile(jnp.zeros(env.model_mj.nu).at[0].set(0.5)[None, :], (batch_size, 1))
    s_left = env.step_static(s_batch, actions_left)
    s_right = env.step_static(s_batch, actions_right)

    shoulders_left = np.asarray(s_left.qpos[:, 7])
    shoulders_right = np.asarray(s_right.qpos[:, 7])
    assert shoulders_left.shape == (batch_size,)
    assert shoulders_right.shape == (batch_size,)
    assert np.all(np.isfinite(shoulders_left)), f"left shoulders should be finite: {shoulders_left}"
    assert np.all(np.isfinite(shoulders_right)), f"right shoulders should be finite: {shoulders_right}"
    assert (shoulders_right > shoulders_left).all(), \
        f"Right shoulders should be > left shoulders: L={shoulders_left}, R={shoulders_right}"


def test_reset_places_arm_in_home_pose_at_correct_qpos_slot():
    """Regression: qpos[0:7] is the cube's freejoint; arm joints live at qpos[7:13].
    Setting home[0:6] must populate the arm joints, not the cube.
    """
    env = UR5eMJMJX()
    home = [0.0, -0.5, 0.9, 0.0, 0.0, 0.0]
    s = env.reset(np.asarray(home))
    # Cube should be at default position (cube rest pos in XML, not home angles)
    cube_xyz = np.asarray(s.qpos[0:3])
    # Arm joint qpos[7:13] should match home[0:6]
    arm_qpos = np.asarray(s.qpos[7:13])
    assert np.allclose(arm_qpos, home, atol=1e-5), \
        f"Arm qpos[7:13]={arm_qpos} should match home={home}"
    # Cube should NOT be at qpos[0:3] = [-1.5708, ...] which was the old bug
    assert not np.any(np.isclose(cube_xyz, -1.5708, atol=0.5)), \
        f"Cube xyz {cube_xyz} should not be at -1.5708 (old bug: home written to cube slot)"


def test_mjx_make_data_batched_init():
    """Proper way per MJX docs: use mjx.make_data inside a vmap for batched envs.

    Verify that batched init via make_data + vmap produces a valid state and
    that running step under vmap drives all envs forward.
    """
    env = UR5eMJMJX()
    batch_size = 8

    @jax.vmap
    def make_vmap_d(_):
        return mjx.make_data(env.model)

    @jax.vmap
    def step_vmap_actions(d, action):
        d = d.replace(ctrl=action)
        return mjx.step(env.model, d)

    # Build indices for vmap
    batched_d = make_vmap_d(jnp.arange(batch_size))
    assert batched_d.qpos.shape[0] == batch_size
    assert batched_d.qpos.shape[1] == env.model_mj.nq

    actions = jnp.zeros((batch_size, env.model_mj.nu)).at[:, 0].set(0.5)
    batched_d_after = step_vmap_actions(batched_d, actions)
    shoulders_after = np.asarray(batched_d_after.qpos[:, 7])
    shoulders_before = np.asarray(batched_d.qpos[:, 7])
    deltas_shoulders = shoulders_after - shoulders_before
    assert np.all(np.isfinite(deltas_shoulders)), \
        f"Shoulder deltas must be finite: {deltas_shoulders}"
    assert (np.abs(deltas_shoulders) > 1e-5).all(), \
        f"Shoulders should move under ctrl=0.5 across all envs: deltas={deltas_shoulders}"
