"""Replay recorded joint targets to validate the data/action contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pyarrow.parquet as pq
from types import SimpleNamespace

from bude_vla.action_space import (
    apply_policy_action,
    make_ik_controller,
    uses_ik_action_space,
)
from bude_vla.envs.so101_mjx import (
    PICK_WORKSPACE_X_RANGE,
    PICK_WORKSPACE_Y_RANGE,
    is_grasping_from_contacts,
    load_arm_model,
)
from eval_pick_ball import is_success, reset_arm, reset_cube


def load_episode(root: Path, meta_path: Path) -> tuple[dict, np.ndarray]:
    meta = json.loads(meta_path.read_text())
    episode_index = int(meta["episode_index"])
    chunk = episode_index // 1000
    parquet = (
        root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    table = pq.read_table(parquet, columns=["action"])
    actions = np.stack([
        np.asarray(value.as_py(), dtype=np.float64) for value in table["action"]
    ])
    return meta, actions


def replay_one(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_xy: tuple[float, float],
    actions: np.ndarray,
    sim_substeps_per_action: int,
    action_space: str = "joint_abs",
) -> dict:
    mujoco.mj_resetData(model, data)
    reset_arm(model, data)
    reset_cube(data, cube_xy[0], cube_xy[1])
    for _ in range(50):
        mujoco.mj_step(model, data)

    ever_grasped = False
    cfg = SimpleNamespace(action_space=action_space, ee_delta_scale=0.05)
    ik = make_ik_controller(model, data) if uses_ik_action_space(cfg) else None
    expected_dim = 4 if uses_ik_action_space(cfg) else model.nu
    for action in actions:
        if action.shape[0] != expected_dim:
            raise ValueError(
                f"{action_space} replay requires {expected_dim}D controls, "
                f"got {action.shape[0]}"
            )
        if uses_ik_action_space(cfg):
            apply_policy_action(model, data, action, cfg, ik=ik)
        else:
            data.ctrl[:] = action
        for _ in range(sim_substeps_per_action):
            mujoco.mj_step(model, data)
            ever_grasped = ever_grasped or (
                is_grasping_from_contacts(model, data) > 0.5
            )

    return {
        "grasped": ever_grasped,
        "success": bool(ever_grasped and is_success(data)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-success-rate", type=float, default=0.90,
                    help="Abort when persisted-action replay success is below this rate.")
    args = ap.parse_args()

    root = Path(args.data_root)
    info = json.loads((root / "meta" / "info.json").read_text())
    default_substeps = int(info.get("sim_substeps_per_action", 16))
    action_space = str(info.get("action_space", "joint_abs"))
    episode_files = sorted((root / "meta" / "episodes_index").glob("*.json"))
    if len(episode_files) > args.num_episodes:
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(
            len(episode_files), size=args.num_episodes, replace=False
        ))
        episode_files = [episode_files[int(i)] for i in selected]
    if not episode_files:
        raise FileNotFoundError(f"no episodes found in {root}")

    model = load_arm_model()
    data = mujoco.MjData(model)
    n_grasp = 0
    n_success = 0
    for path in episode_files:
        meta, actions = load_episode(root, path)
        if "cube_start_xy" not in meta:
            raise ValueError(
                f"{path} lacks cube_start_xy; replay requires camera-fixed v37+ data"
            )
        cube_xy = tuple(float(v) for v in meta["cube_start_xy"])
        if not (
            PICK_WORKSPACE_X_RANGE[0] <= cube_xy[0] <= PICK_WORKSPACE_X_RANGE[1]
            and PICK_WORKSPACE_Y_RANGE[0] <= cube_xy[1] <= PICK_WORKSPACE_Y_RANGE[1]
        ):
            print(f"warning: cube {cube_xy} is outside the validated workspace")
        result = replay_one(
            model,
            data,
            cube_xy,
            actions,
            int(meta.get("sim_substeps_per_action", default_substeps)),
            action_space=action_space,
        )
        n_grasp += int(result["grasped"])
        n_success += int(result["success"])
        print(
            f"ep {meta['episode_index']:04d} frames={len(actions)} "
            f"cube=({cube_xy[0]:.3f},{cube_xy[1]:.3f}) "
            f"grasp={result['grasped']} success={result['success']}",
            flush=True,
        )

    n = len(episode_files)
    print("\n=== DATASET REPLAY ===")
    print(f"episodes: {n}")
    print(f"strict grasp: {n_grasp}/{n} ({n_grasp / n:.3f})")
    print(f"success: {n_success}/{n} ({n_success / n:.3f})")
    success_rate = n_success / n
    if success_rate < args.min_success_rate:
        raise SystemExit(
            f"replay success {success_rate:.3f} below required {args.min_success_rate:.3f}; do not train"
        )


if __name__ == "__main__":
    main()
