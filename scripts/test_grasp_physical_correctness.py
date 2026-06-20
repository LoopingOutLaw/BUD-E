"""Test gate that locks the floor-penetration bug and the drive-by-grasp bug.

This is not a unittest-style test. It's a deterministic harness with two
literal assertions:

  Assertion A (floor penetration): at the step where `grasp.attached`
    first flips True, the gripper body must NOT be in contact with the
    world/floor/bowl-floor geometry.  (The "arm passes through table"
    bug is what this guards.)

  Assertion B (real grasp): at the step where `grasp.attached` first
    flips True, `moving_jaw ↔ cube` must be one of the active contacts.
    (The "drive-by" attach via static-frame grazing must not count.)

If either assertion fails for any of the N attempts, the script exits 1.
Otherwise it exits 0 and prints a per-attempt summary.

Red-phase reproduction: run on commit 9bf50b2 (V1 baseline) — should
print FLOOR_PENETRATION and/or DRIVE_BY_ATTACH verdicts.
Green-phase: run on the fix — must print PASS on every attempt.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import mujoco
import numpy as np

from bude_vla.envs.so101_mjx import load_arm_model, default_joint_angles, CUBE_QPOS_START
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace
from bude_vla.grasp import GraspController


def run_one(model, cx, cy):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:5] = default_joint_angles(model)
    data.qpos[5] = 1.5
        data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [cx, cy, 0.010]  # cube half-extent on world floor
    data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    plan = ScriptedPickAndPlace(model, data, cube_start_xy=np.array([cx, cy]))
    verdict = {
        "attached_step": None,
        "gripper_penetrates_floor": None,
        "jaw_ball_contact_present": None,
        "all_contacts_at_attach": None,
    }

    for step in range(600):
        ctrl, _, done, info = plan.step(model, data)
        data.ctrl[:] = ctrl
        for _ in range(5):
            mujoco.mj_step(model, data)

        if info.get("attached") and verdict["attached_step"] is None:
            verdict["attached_step"] = step
            active = []
            for i in range(data.ncon):
                c = data.contact[i]
                b1 = model.geom_bodyid[c.geom1]
                b2 = model.geom_bodyid[c.geom2]
                n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or "world"
                n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or "world"
                active.append(f"{n1}↔{n2}")
            verdict["all_contacts_at_attach"] = active
            verdict["gripper_penetrates_floor"] = any(
                ("gripper" in s and ("world" in s or "table" in s or "bowl" in s.lower() or "floor" in s.lower()))
                for s in active
            )
            verdict["jaw_ball_contact_present"] = any(
                ("moving_jaw" in s and "cube" in s) for s in active
            )
            break
        if done:
            break
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--attempts", type=int, default=5)
    args = ap.parse_args()

    model = load_arm_model()
    rng = np.random.default_rng(args.seed)

    summary = []
    fail_count = 0
    for a in range(args.attempts):
        cx = float(rng.uniform(0.285, 0.315))
        cy = float(rng.uniform(-0.015, 0.015))
        v = run_one(model, cx, cy)
        summary.append({"attempt": a + 1, "cx": cx, "cy": cy, **v})
        if v["attached_step"] is None:
            flag = "FAIL_NO_ATTACH"
            fail_count += 1
        elif v["gripper_penetrates_floor"]:
            flag = "FAIL_FLOOR_PENETRATION"
            fail_count += 1
        elif not v["jaw_ball_contact_present"]:
            flag = "FAIL_DRIVE_BY"
            fail_count += 1
        else:
            flag = "PASS"
        print(f"attempt {a+1}: cx={cx:.3f} cy={cy:+.3f}  attached={v['attached_step']}  "
              f"penetrates_floor={v['gripper_penetrates_floor']}  "
              f"jaw_ball={v['jaw_ball_contact_present']}  -> {flag}")

    print(f"\nTotal fails: {fail_count}/{args.attempts}")
    if fail_count:
        print("VERDICT: BUG_PRESENT")
        sys.exit(1)
    else:
        print("VERDICT: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
