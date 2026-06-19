"""CPU-only MuJoCo recorder: collect episodes without JAX GPU allocation.

Uses pure mujoco.MjModel / mujoco.MjData for physics and mujoco.Renderer
for images, leaving GPU memory free for EGL rendering.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import mujoco
import numpy as np

from bude_vla.envs.so101_mjx import (
    ARM_QPOS_START, ARM_QPOS_END,
    GRIPPER_QPOS_START, GRIPPER_QPOS_END,
    CUBE_QPOS_START, CUBE_QPOS_END, load_arm_model,
)
from bude_vla.data.scripted_policies import scripted_push_step
from bude_vla.data.lerobot_v3 import write_episode


INSTRUCTION_BY_TASK = {
    "reach": "reach the red target",
    "push": "push the cube to the green zone",
    "pick": "pick the cube and place at the target",
}


class CPURecorder:

    def __init__(self, xml_path: str | Path | None = None):
        if xml_path is not None:
            self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        else:
            self.model = load_arm_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=64, width=64)
        self.nu = self.model.nu
        self.nq = self.model.nq

    def _reset(self, qpos: np.ndarray | None = None,
               cube_xyz: tuple[float, float, float] = (0.6, 0.0, 0.445)):
        mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            n = min(len(qpos), self.nq)
            self.data.qpos[:n] = qpos[:n]
        self.data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = cube_xyz
        self.data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)

    def _render(self) -> np.ndarray:
        self.renderer.update_scene(self.data)
        return self.renderer.render().copy()

    def collect_reach(self, target_xyz: np.ndarray, n_steps: int = 30,
                      seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        home = np.zeros(self.nq, dtype=np.float64)
        home[ARM_QPOS_START:ARM_QPOS_END] = [0.0, -0.5, 0.95, -0.55, 0.0]
        self._reset(home)

        images: List[np.ndarray] = []
        qposes: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        rewards_sum = 0.0
        success = False

        for t in range(n_steps):
            img = self._render()
            qpos = self.data.qpos.copy()
            ee = self.data.site_xpos[0].copy()

            delta = target_xyz - ee
            action = np.zeros(self.nu, dtype=np.float32)
            if self.nu >= 3:
                action[0] = np.clip(-delta[1] * 4.0, -1, 1)
                action[1] = np.clip(-delta[0] * 4.0, -1, 1)
                action[2] = np.clip(delta[2] * 2.0, -1, 1)

            rewards_sum += -float(np.linalg.norm(ee - target_xyz))
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)

            images.append(img)
            qposes.append(qpos)
            actions.append(action)

            if np.linalg.norm(ee - target_xyz) < 0.04:
                success = True
                break

        return {
            "instruction": INSTRUCTION_BY_TASK["reach"],
            "images": np.stack(images),
            "qpos": np.stack(qposes),
            "proprio": np.stack(qposes)[:, ARM_QPOS_START:GRIPPER_QPOS_END].astype(np.float32),
            "actions": np.stack(actions),
            "total_reward": rewards_sum,
            "success": success,
        }

    def collect_push(self, target_2d: np.ndarray, n_steps: int = 40,
                     seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        cube_start_y = rng.uniform(-0.15, 0.15)

        cube_x_init = 0.6
        qpos = np.zeros(self.nq, dtype=np.float64)
        qpos[ARM_QPOS_START:ARM_QPOS_END] = [0.0, -0.5, 0.95, -0.55, 0.0]
        qpos[1] = -0.5
        qpos[CUBE_QPOS_START] = cube_x_init
        qpos[CUBE_QPOS_START + 1] = cube_start_y
        qpos[CUBE_QPOS_START + 2] = 0.445
        qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
        self._reset(qpos, cube_xyz=(cube_x_init, cube_start_y, 0.445))

        target_body_id = next(i for i in range(self.model.nbody)
                              if self.model.body(i).name == "target_zone")
        target_pos_static = self.model.body_pos[target_body_id].copy()
        target_3d = np.array([target_pos_static[0] + target_2d[0],
                              target_pos_static[1] + target_2d[1],
                              target_pos_static[2]], dtype=np.float32)

        cube_body_id = next(i for i in range(self.model.nbody)
                            if self.model.body(i).name == "cube")

        images: List[np.ndarray] = []
        qposes: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        phase = 0
        success = False
        rewards_sum = 0.0

        for t in range(n_steps):
            img = self._render()
            qpos_now = self.data.qpos.copy()
            ee = self.data.site_xpos[0].copy()
            cube_pos = self.data.xpos[cube_body_id].copy()

            action, phase = scripted_push_step(ee, cube_pos, target_3d, phase, nu=self.nu)
            action[-1] = -0.6

            rewards_sum += -float(np.linalg.norm(cube_pos - target_3d))
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)

            images.append(img)
            qposes.append(qpos_now)
            actions.append(action)

            if np.linalg.norm(cube_pos[:2] - target_3d[:2]) < 0.05:
                success = True
                break

        return {
            "instruction": INSTRUCTION_BY_TASK["push"],
            "images": np.stack(images),
            "qpos": np.stack(qposes),
            "proprio": np.stack(qposes)[:, ARM_QPOS_START:GRIPPER_QPOS_END].astype(np.float32),
            "actions": np.stack(actions),
            "total_reward": rewards_sum,
            "success": success,
        }


def record_dataset_cpu(task: str = "reach", n_episodes: int = 100,
                       n_steps: int = 30, root: str | Path = "data/lerobot_v3",
                       xml_path: str | Path | None = None) -> Path:
    """Record n_episodes using CPU-only MuJoCo (no JAX GPU memory)."""
    rec = CPURecorder(xml_path)
    rng = np.random.default_rng(0)
    for i in range(n_episodes):
        if task == "reach":
            target = rng.uniform([0.4, -0.3, 0.42], [0.8, 0.3, 0.65], size=3).astype(np.float32)
            ep = rec.collect_reach(target, n_steps=n_steps, seed=i)
        elif task == "push":
            target_2d = rng.uniform([-0.1, -0.15], [0.1, 0.15], size=2).astype(np.float32)
            ep = rec.collect_push(target_2d, n_steps=n_steps, seed=i)
        else:
            raise ValueError(f"Unknown task: {task}")
        write_episode(root, ep)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{task}] episode {i + 1}/{n_episodes} done")
    return Path(root)
