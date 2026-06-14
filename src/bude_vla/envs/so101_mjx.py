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
    """The 'home' config the arm resets to."""
    home = np.asarray([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    if len(home) < model.njnt:
        home = np.concatenate([home, np.zeros(model.njnt - len(home))])
    return home[: model.njnt]


class UR5eMJMJX:
    """PyTorch-free interface; everything is JAX/jnp so we can vmap / jit.

    Naming preserves `mujoco.mjx` semantics where useful.
    """

    def __init__(self, xml_path: str | Path | None = None):
        self.model_mj = load_arm_model(xml_path)
        self.model = mjx.put_model(self.model_mj)
        self.n_arm = 6
        # Number of actuators is whatever MJX reports
        self.nu = self.model_mj.nu
        self.action_dim = self.nu
        self.n_qpos = self.model_mj.nq
        self.n_qvel = self.model_mj.nv

    def make_data(self, joint_angles: np.ndarray | None = None) -> mjx.Data:
        data = mujoco.MjData(self.model_mj)
        if joint_angles is not None:
            data.qpos[: len(joint_angles)] = joint_angles
        mujoco.mj_forward(self.model_mj, data)
        return mjx.put_data(self.model_mj, data)

    def reset(self, joint_angles: np.ndarray | None = None) -> mjx.Data:
        d = self.make_data(joint_angles)
        return d

    @staticmethod
    def _to_action(action) -> jnp.ndarray:
        a = jnp.asarray(action, dtype=jnp.float32)
        if a.ndim == 1:
            a = a[None, :]
        return a

    def step_static(self, state: mjx.Data, action) -> mjx.Data:
        """Apply action[0] (single env step) and return new state."""
        a = self._to_action(action)[0]
        # If action dim has 7 entries and there are 7 actuators, ctrl = a directly
        n_controls = self.model_mj.nu
        if a.shape[0] != n_controls:
            raise ValueError(
                f"Action dim {a.shape[0]} != model.nu {n_controls}. "
                f"Action order must match the XML's actuator order."
            )
        new = state.replace(ctrl=a)
        new = mjx.step(self.model, new)
        return new

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
