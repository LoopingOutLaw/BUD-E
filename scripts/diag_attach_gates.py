"""Diagnostic instrumentation: run one episode and log jaw_qpos / gap / contact
during the GRASP window so we can see which attach gate is failing.
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mujoco
import numpy as np

from bude_vla.envs.so101_mjx import (
    load_arm_model, default_joint_angles,
    ARM_QPOS_START, GRIPPER_QPOS_START, CUBE_QPOS_START, CUBE_QPOS_END,
)
from bude_vla.grasp import (
    GraspController, BALL_RADIUS, ATTACH_GAP_TOLERANCE, ATTACH_DEBOUNCE_STEPS,
    JAW_CLOSED_QPOS_THRESHOLD,
)
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace


def run_one(seed: int, ep_idx: int) -> None:
    model = load_arm_model()
    rng = np.random.default_rng(seed)
    cx = float(rng.uniform(0.28, 0.32))
    cy = float(rng.uniform(-0.02, 0.02))

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:5] = default_joint_angles(model)
    data.qpos[5] = 1.5
    data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [cx, cy, 0.0295]
    data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array([cx, cy]))
    print(f"\n=== ep {ep_idx} cx={cx:.3f} cy={cy:.3f} ===")
    print(f"  thresholds: gap<={ATTACH_GAP_TOLERANCE*1000:.1f}mm  "
          f"jaw<={JAW_CLOSED_QPOS_THRESHOLD:.2f}  "
          f"debounce={ATTACH_DEBOUNCE_STEPS}")
    print(f"  {'step':>4} {'phase':>5} {'phase_st':>7} {'jaw_qpos':>8} "
          f"{'gap_mm':>8} {'contact':>7} {'streak':>6} {'attached':>8} {'dist_mm':>8}")

    ball_start_xyz = data.xpos[policy.cube_body_id].copy()

    for step in range(600):
        ctrl, arm_target, done, info = policy.step(model, data)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if step % 5 == 0 or info.get("attached") or step < 5 or step > 540:
            jaw_qpos = float(data.qpos[GRIPPER_QPOS_START])
            gap = policy.grasp.gap(data) * 1000  # mm
            contact = policy.grasp._has_jaw_ball_contact(model, data)
            streak = policy.grasp.state.enclosure_streak
            attached = policy.grasp.state.attached
            ball_xyz = data.xpos[policy.cube_body_id].copy()
            drift = float(np.linalg.norm(ball_xyz - ball_start_xyz)) * 1000
            print(f"  {step:4d} {info['phase']:5d} {info['phase_step']:7d} "
                  f"{jaw_qpos:8.3f} {gap:8.2f} {int(contact):7d} "
                  f"{streak:6d} {int(attached):8d} {drift:8.1f}")

        if done:
            break


if __name__ == "__main__":
    seed = int(os.environ.get("SEED", "0"))
    for i in range(3):
        run_one(seed + i, i)
