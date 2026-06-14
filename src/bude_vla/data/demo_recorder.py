"""Demo recorder: collects (image, qpos, instruction, action) tuples from a BUD-E env.

For the v1 demo recorder we use the MJX CPU-side MuJoCo backend because MJX
GPU doesn't trivially batch image renders without GPU buffer plumbing. We
record on CPU for a few episodes per task. Phase II can move this to GPU+mujoco_warp.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from bude_vla.envs.so101_mjx import UR5eMJMJX


INSTRUCTION_BY_TASK = {
    "reach": "reach the red target",
    "push": "push the cube to the green zone",
    "pick": "pick the cube and place at the target",
}


def scripted_reach_step(ee: np.ndarray, target: np.ndarray, qpos: np.ndarray,
                         ctrl_lo: np.ndarray, ctrl_hi: np.ndarray,
                         nu: int = 7) -> np.ndarray:
    """PD-style control: move EE toward target by adjusting the first 3 arm joints."""
    delta = target - ee
    action = np.zeros(nu, dtype=np.float32)
    if nu >= 3:
        action[0] = np.clip(-delta[1] * 4.0, -1, 1)
        action[1] = np.clip(-delta[0] * 4.0, -1, 1)
        action[2] = np.clip(delta[2] * 2.0, -1, 1)
    return action


def collect_reach_episode(env: UR5eMJMJX, target_xyz: np.ndarray,
                           n_steps: int = 30, seed: int = 0) -> dict:
    """One reach demonstration episode."""
    rng = np.random.default_rng(seed)

    # Set the arm to a near-target config and put a free-joint target sphere near
    # the start, then run the scripted policy
    s = env.reset(np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    images: List[np.ndarray] = []
    qposes: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards_sum = 0.0
    success = False

    ctrl_lo, ctrl_hi = env.action_bounds()

    for t in range(n_steps):
        # Render current state
        img = env.render(s, height=64, width=64)
        qpos = np.asarray(s.qpos, dtype=np.float32)
        ee = np.asarray(s.site_xpos[0], dtype=np.float32)  # ee_center site

        # Pick action (PD-style)
        action = scripted_reach_step(ee, target_xyz, qpos, ctrl_lo, ctrl_hi, nu=env.model_mj.nu)
        rewards_sum += -float(np.linalg.norm(ee - target_xyz))

        s_new = env.step_static(s, action)
        s = s_new
        images.append(img)
        qposes.append(qpos)
        actions.append(action)

        # Check success (EE close to target)
        if np.linalg.norm(ee - target_xyz) < 0.04:
            success = True
            break

    return {
        "instruction": INSTRUCTION_BY_TASK["reach"],
        "images": np.stack(images),         # (T, 64, 64, 3)
        "qpos": np.stack(qposes),           # (T, njnt)
        "actions": np.stack(actions),       # (T, nu)
        "total_reward": rewards_sum,
        "success": success,
    }


def save_episode_npz(episode: dict, out_dir: str | Path) -> str:
    """Save episode to npz for fast iteration during TDD and v1 evaluation.

    LeRobotDataset v3 conversion (Parquet+MP4) comes in Task 12.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = len(list(out_dir.glob("episode_*.npz")))
    path = out_dir / f"episode_{idx:04d}.npz"
    np.savez(
        path,
        instruction=episode["instruction"],
        images=episode["images"],
        qpos=episode["qpos"],
        actions=episode["actions"],
        total_reward=episode["total_reward"],
        success=episode["success"],
    )
    return str(path)
