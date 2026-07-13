"""Benchmark the scripted pick expert on random cube positions.

This measures the simulator/expert ceiling independently of learned policy
quality. The expert may use simulator state; this is not a policy input path.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from bude_vla.envs.so101_mjx import (
    CUBE_QPOS_START,
    CUBE_QPOS_END,
    CUBE_REST_Z,
    GRIPPER_QPOS_START,
    EXPERT_CONTROL_SUBSTEPS,
    PICK_WORKSPACE_X_RANGE,
    PICK_WORKSPACE_Y_RANGE,
    BowlPlacementTracker,
    is_grasping_from_contacts,
    is_touching_cube_from_contacts,
    load_arm_model,
)
from bude_vla.scripted_pick_and_place import GRIPPER_OPEN, ScriptedPickAndPlace, WRIST_FLEX_LOCK, WRIST_ROLL_LOCK
from eval_pick_ball import parse_cube_positions

SUBSTEPS_PER_FRAME = EXPERT_CONTROL_SUBSTEPS
def reset_arm(model, data) -> None:
    data.qpos[0] = 0.0
    data.qpos[1] = -0.5
    data.qpos[2] = 0.95
    data.qpos[3] = WRIST_FLEX_LOCK
    data.qpos[4] = WRIST_ROLL_LOCK
    data.qpos[GRIPPER_QPOS_START] = GRIPPER_OPEN
    data.ctrl[:5] = data.qpos[:5]
    data.ctrl[GRIPPER_QPOS_START] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)


def reset_cube(data, cx: float, cy: float) -> None:
    data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [cx, cy, CUBE_REST_Z]
    data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[CUBE_QPOS_START:CUBE_QPOS_END] = 0.0
    mujoco.mj_forward(data.model, data)


def run_one(model, data, cube_xy: tuple[float, float], max_steps: int) -> dict:
    mujoco.mj_resetData(model, data)
    reset_arm(model, data)
    reset_cube(data, cube_xy[0], cube_xy[1])
    for _ in range(50):
        mujoco.mj_step(model, data)

    expert = ScriptedPickAndPlace(model, data, cube_xy, max_grasp_retries=1)
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    touch_frames = 0
    grasp_frames = 0
    ever_grasped = False
    placement = BowlPlacementTracker()
    placed = False
    done_steps = 0

    for step in range(max_steps):
        ctrl, _arm_q, done, _info = expert.step(model, data)
        data.ctrl[:] = ctrl
        for _ in range(SUBSTEPS_PER_FRAME):
            mujoco.mj_step(model, data)
        touching = is_touching_cube_from_contacts(model, data) > 0.5
        grasping = is_grasping_from_contacts(model, data) > 0.5
        touch_frames += int(touching)
        grasp_frames += int(grasping)
        ever_grasped = ever_grasped or grasping
        placed = placement.update(model, data)
        if done:
            done_steps += 1
            if placed or done_steps >= placement.required_steps:
                break

    cube = data.xpos[cube_id].copy()
    success = bool(ever_grasped and placed)
    return {
        "success": success,
        "touch_frames": touch_frames,
        "grasp_frames": grasp_frames,
        "steps": step + 1,
        "cube_final": cube,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=2200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cube-positions", default=None)
    ap.add_argument("--cube-x-range",
                    default=",".join(str(v) for v in PICK_WORKSPACE_X_RANGE))
    ap.add_argument("--cube-y-range",
                    default=",".join(str(v) for v in PICK_WORKSPACE_Y_RANGE))
    args = ap.parse_args()

    model = load_arm_model()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)
    fixed_positions = parse_cube_positions(args.cube_positions)
    x_range = tuple(float(v) for v in args.cube_x_range.split(","))
    y_range = tuple(float(v) for v in args.cube_y_range.split(","))

    results = []
    for ep in range(args.num_episodes):
        if fixed_positions:
            cube_xy = fixed_positions[ep % len(fixed_positions)]
        else:
            cube_xy = (float(rng.uniform(*x_range)), float(rng.uniform(*y_range)))
        result = run_one(model, data, cube_xy, args.max_steps)
        results.append(result)
        print(
            f"ep {ep:03d} cube=({cube_xy[0]:.3f},{cube_xy[1]:.3f}) "
            f"touch={result['touch_frames']} grasp={result['grasp_frames']} "
            f"success={result['success']} steps={result['steps']}",
            flush=True,
        )

    n = len(results)
    successes = sum(int(r["success"]) for r in results)
    touched = sum(int(r["touch_frames"] > 0) for r in results)
    grasped = sum(int(r["grasp_frames"] > 0) for r in results)
    print("\n=== SCRIPTED PICK BENCH ===")
    print(f"episodes: {n}")
    print(f"success episodes: {successes}/{n} ({successes / n:.3f})")
    print(f"any_contact episodes: {touched}/{n} ({touched / n:.3f})")
    print(f"strict_grasp episodes: {grasped}/{n} ({grasped / n:.3f})")
    print(f"any_contact frames: {sum(int(r['touch_frames']) for r in results)}")
    print(f"strict_grasp frames: {sum(int(r['grasp_frames']) for r in results)}")


if __name__ == "__main__":
    main()
