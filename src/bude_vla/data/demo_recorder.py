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
from bude_vla.data.scripted_policies import scripted_push_step
from bude_vla.data.lerobot_v3 import write_episode


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
        "qpos": np.stack(qposes),           # (T, njnt) full state incl cube
        "proprio": np.stack(qposes)[:, 7:15],  # (T, 8) arm+gripper only
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


def collect_push_episode(env, target_2d: np.ndarray, n_steps: int = 40,
                          seed: int = 0,
                          start_qpos: np.ndarray | None = None) -> dict:
    """One push demonstration episode.

    Cube starts at (0.6, 0, 0.445). target_2d is the (x, y) of the target zone.
    Camera; policy pushes the cube toward target_2d.
    """
    rng = np.random.default_rng(seed)
    # Randomize cube start position
    cube_start_y = rng.uniform(-0.15, 0.15)
    if start_qpos is None:
        start_qpos = np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Set qpos so first 7 entries are arm joints+gripper; positions 7+ are freejoint of cube
    nq = env.model_mj.nq
    qpos = np.zeros(nq, dtype=np.float64)
    qpos[:8] = np.concatenate([start_qpos[:7], [cube_start_y]]) if nq >= 8 else start_qpos[:nq]
    # If there are positions for the cube freejoint (7 of them: 3 pos + 4 quat), set them
    # qpos layout: [arm 6 joints + 1 gripper, cube freejoint 7 (xyz, quat)]. Total = 14.
    # Above I only filled positions 0..7. Need cube xyz at pos 8..10 and quat 11..14.
    if nq >= 15:
        qpos[8] = 0.6
        qpos[9] = cube_start_y
        qpos[10] = 0.445
        qpos[11:15] = [1.0, 0.0, 0.0, 0.0]  # unit quat

    s = env.reset(joint_angles=qpos.astype(np.float32))

    # Find the cube and target body indices by name scan
    cube_body_id = next(i for i in range(env.model_mj.nbody)
                         if env.model_mj.body(i).name == "cube")
    target_body_id = next(i for i in range(env.model_mj.nbody)
                          if env.model_mj.body(i).name == "target_zone")
    target_pos_static = np.asarray(env.model_mj.body_pos[target_body_id], dtype=np.float32)
    target_2d_full = np.array([target_pos_static[0] + target_2d[0],
                                target_pos_static[1] + target_2d[1],
                                target_pos_static[2]],
                               dtype=np.float32)

    images: List[np.ndarray] = []
    qposes: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    phase = 0
    success = False
    rewards_sum = 0.0

    cube_id = cube_body_id
    for t in range(n_steps):
        img = env.render(s, height=64, width=64)
        qpos_now = np.asarray(s.qpos, dtype=np.float32)
        ee = np.asarray(s.site_xpos[0], dtype=np.float32)
        cube_pos = np.asarray(s.xpos[cube_id], dtype=np.float32)
        action, phase = scripted_push_step(ee, cube_pos, target_2d_full,
                                            phase, nu=env.model_mj.nu)
        # On reaching phase 1, open gripper (gripper is finger_left, +collapse pulls fingers in,
        # so -1 opens fully — qpos=0 here is the default middle; we just drive toward -0.04)
        action[-1] = -0.6  # open gripper fully via negative ctrl

        rewards_sum += -float(np.linalg.norm(cube_pos - target_2d_full))

        s_new = env.step_static(s, action)
        s = s_new
        images.append(img)
        qposes.append(qpos_now)
        actions.append(action)

        if np.linalg.norm(cube_pos[:2] - target_2d_full[:2]) < 0.05:
            success = True
            break

    return {
        "instruction": INSTRUCTION_BY_TASK["push"],
        "images": np.stack(images),
        "qpos": np.stack(qposes),
        "proprio": np.stack(qposes)[:, 7:15],
        "actions": np.stack(actions),
        "total_reward": rewards_sum,
        "success": success,
    }


def record_dataset(env: UR5eMJMJX, task: str = "reach", n_episodes: int = 100,
                    n_steps: int = 30, root: str | Path = "data/lerobot_v3") -> Path:
    """Record n_episodes of a task and write them to LeRobot v3 layout.

    Args:
        env: UR5eMJMJX instance.
        task: "reach" or "push".
        n_episodes: number of episodes to record.
        n_steps: max steps per episode.
        root: output directory for the v3 dataset.

    Returns:
        Path to the dataset root.
    """
    rng = np.random.default_rng(0)
    for i in range(n_episodes):
        if task == "reach":
            target = rng.uniform([0.4, -0.3, 0.42], [0.8, 0.3, 0.65], size=3).astype(np.float32)
            ep = collect_reach_episode(env, target, n_steps=n_steps, seed=i)
        elif task == "push":
            target_2d = rng.uniform([-0.1, -0.15], [0.1, 0.15], size=2).astype(np.float32)
            ep = collect_push_episode(env, target_2d, n_steps=n_steps, seed=i)
        else:
            raise ValueError(f"Unknown task: {task}")
        write_episode(root, ep)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{task}] episode {i + 1}/{n_episodes} done")
    return Path(root)
