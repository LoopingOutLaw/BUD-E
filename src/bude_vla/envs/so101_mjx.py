"""Wrapper around MuJoCo/MJX for BUD-E's UR5e-style 6+1 DOF arm.

For Task 9 we use MuJoCo's bundled `arm26.xml` (which ships with meshed visuals
of an arm6 robot, see mujoco/mjx/test_data/) under `ur5e/scene.xml`.

We re-export a few MuJoCo utilities, attach a parallel-jaw gripper (we treat
joints_actuator[6] as gripper open/close), and provide:
    reset(joint_angles) -> mjx.Data
    step(action) -> (mjx.Data, reward, done, info)

This module is the *single* interface between the rest of BUD-E and physics.
Tasks in `bude_vla.envs.tasks` build on top of `UR5eMJMJX`.

Joint order: 6 arm joints + 1 gripper (binary). Action: 7D continuous.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx


ARM_MODEL_PATH = Path(__file__).resolve().parents[3] / "urdf" / "ur5e_scene.xml"


def load_arm_model(xml_path: str | Path | None = None) -> mujoco.MjModel:
    """Load arm + scene MJCF. Defaults to the bundled UR5e scene file."""
    path = Path(xml_path) if xml_path else ARM_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Arm scene not found at {path}")
    return mujoco.MjModel.from_xml_string(path.read_text())


def default_joint_angles(model: mujoco.MjModel) -> np.ndarray:
    """The 'home' config the arm resets to (6 arm joints, in joint-ID order)."""
    home = np.asarray([0.0, -0.5, 0.9, 0.0, 0.0, 0.0])
    if len(home) < model.njnt:
        home = np.concatenate([home, np.zeros(model.njnt - len(home))])
    return home[: model.njnt]


# qpos layout for ur5e_scene.xml:
#   qpos[0:7] = cube freejoint (xyz + quat wxyz)
#   qpos[7:13] = 6 arm joints (shoulder_pan, shoulder_lift, elbow, forearm_roll, wrist_pitch, wrist_roll)
#   qpos[13:14] = finger_left slider
#   qpos[14:15] = finger_right slider (mirror of finger_left via equality)
ARM_QPOS_START = 7
ARM_QPOS_END = 13
CUBE_QPOS_END = 7
GRIPPER_QPOS_START = 13
GRIPPER_QPOS_END = 15


class UR5eMJMJX:
    """PyTorch-free interface; everything is JAX/jnp so we can vmap / jit.

    Naming preserves `mujoco.mjx` semantics where useful.
    """

    def __init__(self, xml_path: str | Path | None = None):
        self.model_mj = load_arm_model(xml_path)
        self.model = mjx.put_model(self.model_mj)
        self.n_arm = 6
        self.nu = self.model_mj.nu
        self.action_dim = self.nu
        self.n_qpos = self.model_mj.nq
        self.n_qvel = self.model_mj.nv

    def make_data(self, joint_angles: np.ndarray | None = None,
                  cube_xyz: tuple[float, float, float] = (0.6, 0.0, 0.445)
                  ) -> mjx.Data:
        """Reset MJX data.

        Args:
            joint_angles: 6-vector of arm joint angles. Goes to qpos[7:13].
                          Optional; if None, uses XML defaults.
            cube_xyz: cube position. Default (0.6, 0, 0.445).
        """
        # Use mjx.make_data (correct per MJX docs for batched use)
        d = mjx.make_data(self.model)
        if joint_angles is not None:
            angles = jnp.asarray(joint_angles, dtype=jnp.float32)
            # Pad to 6 if shorter, then place at qpos[7:13]
            if angles.shape[0] < self.n_arm:
                pad = jnp.zeros(self.n_arm - angles.shape[0], dtype=jnp.float32)
                angles = jnp.concatenate([angles, pad])
            angles = angles[: self.n_arm]
            d = d.replace(qpos=d.qpos.at[ARM_QPOS_START:ARM_QPOS_END].set(angles))
        # Cube xyz
        cube = jnp.asarray(cube_xyz, dtype=jnp.float32)
        d = d.replace(qpos=d.qpos.at[0:3].set(cube))
        # Cube identity quaternion (w, x, y, z)
        d = d.replace(qpos=d.qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)))
        return d

    def reset(self, joint_angles: np.ndarray | None = None,
              cube_xyz: tuple[float, float, float] = (0.6, 0.0, 0.445)
              ) -> mjx.Data:
        return self.make_data(joint_angles, cube_xyz)

    @staticmethod
    def _to_action(action) -> jnp.ndarray:
        a = jnp.asarray(action, dtype=jnp.float32)
        if a.ndim == 1:
            a = a[None, :]
        return a

    def step_static(self, state: mjx.Data, action) -> mjx.Data:
        """Apply action and step. Action can be 1D (single env) or 2D (batched).

        MJX requires `mjx.step` to be vmapped for batched execution (it doesn't
        auto-vectorize a stacked qpos); we detect that here.
        """
        a = self._to_action(action)
        if a.ndim != 2:
            raise ValueError(f"action must be 1D or 2D, got shape {a.shape}")
        n_controls = self.model_mj.nu
        if a.shape[-1] != n_controls:
            raise ValueError(
                f"Action last-dim {a.shape[-1]} != model.nu {n_controls}. "
                f"Action order must match the XML's actuator order."
            )
        is_batched = state.qpos.ndim > 1
        if not is_batched:
            # 1D state: use single-env path
            new = state.replace(ctrl=a[0])
            return mjx.step(self.model, new)
        # Batched state: must vmapped mjx.step, and the action's batch must
        # match the state's leading batch dim.
        if a.shape[0] != state.qpos.shape[0]:
            raise ValueError(
                f"action batch {a.shape[0]} != state batch {state.qpos.shape[0]}"
            )

        @jax.vmap
        def _vmap_step(d, act):
            d = d.replace(ctrl=act)
            return mjx.step(self.model, d)

        return _vmap_step(state, a)

    def jitted_step(self):
        jit_step = jax.jit(lambda state, action: self.step_static(state, action))
        return jit_step

    def render(self, state: mjx.Data, height: int = 224, width: int = 224) -> np.ndarray:
        """Render the current single-env state to an RGB image.

        Note: MJX rendering goes through the underlying mujoco renderer. This is
        mainly for sanity / debug; bulk demos should use a batched renderer.
        """
        d = mjx.get_data(self.model_mj, state)
        try:
            if not hasattr(self, "_renderer") or self._renderer is None:
                self._renderer = mujoco.Renderer(self.model_mj, height=height, width=width)
            else:
                self._renderer._height = height
                self._renderer._width = width
            self._renderer.update_scene(d)
            img = self._renderer.render()
            return img
        except Exception:
            # Headless / no GL context: return a placeholder of the right shape
            return np.zeros((height, width, 3), dtype=np.uint8)

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        ctrl_low = self.model_mj.actuator_ctrlrange[:, 0]
        ctrl_high = self.model_mj.actuator_ctrlrange[:, 1]
        return np.asarray(ctrl_low), np.asarray(ctrl_high)
