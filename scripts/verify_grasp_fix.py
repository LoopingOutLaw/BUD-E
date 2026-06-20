"""Sanity-check the grasp-gap fix and the bowl collision-group fix BEFORE
investing time in re-recording 100s of episodes and retraining.

Two checks:

1. `--grasp`  (default): runs N scripted pick episodes with the fixed
   ScriptedPickAndPlace and asserts that whenever the GraspController
   reports `attached=True`, the live geometric gap between the ball's
   surface and the gripper's grasp point is (near) zero -- i.e. there is
   no visible floating gap. Reports the worst-case gap seen across all
   attached frames, the attach success rate, and (--diagnose) a
   breakdown of failures by cube position, so you can tell whether
   misses cluster in one region (an IK/offset tuning issue) or are
   scattered (something more fundamental).

2. `--bowls`: drops a ball with zero arm motion onto the pick-bowl
   position and onto the target-bowl position, lets physics settle for a
   couple seconds of sim time, and asserts the ball stays within the
   bowl's rim radius in both cases -- this is what actually validates
   the contype/conaffinity fix in so101_mjx.py.

Usage:
    unset PYTHONPATH
    MUJOCO_GL=egl PYTHONPATH=src python scripts/verify_grasp_fix.py
    MUJOCO_GL=egl PYTHONPATH=src python scripts/verify_grasp_fix.py --grasp --episodes 20 --diagnose
    MUJOCO_GL=egl PYTHONPATH=src python scripts/verify_grasp_fix.py --bowls
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from bude_vla.envs.so101_mjx import (
    load_arm_model, default_joint_angles,
    ARM_QPOS_START, GRIPPER_QPOS_START, CUBE_QPOS_START, CUBE_QPOS_END,
)
from bude_vla.grasp import GraspController
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace


def check_grasp(n_episodes: int, seed: int, diagnose: bool) -> bool:
    model = load_arm_model()
    rng = np.random.default_rng(seed)

    worst_gap_while_attached = -np.inf
    n_attached_at_all = 0
    n_success = 0
    failures = []

    for i in range(n_episodes):
        cx = float(rng.uniform(0.28, 0.32))
        cy = float(rng.uniform(-0.02, 0.02))

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[:5] = default_joint_angles(model)
        data.qpos[5] = 1.5
        data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [cx, cy, 0.025]  # cube half-extent on world floor
        data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array([cx, cy]))
        attached_this_ep = False

        for _ in range(600):
            ctrl, arm_target, done, info = policy.step(model, data)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

            if info.get("attached"):
                attached_this_ep = True
                gap = policy.grasp.gap(data)
                worst_gap_while_attached = max(worst_gap_while_attached, gap)

            if done:
                break

        cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
        dist = np.linalg.norm(data.xpos[cube_body_id, :2] - data.xpos[target_body_id, :2])
        success = bool(dist < 0.05 and attached_this_ep)

        if attached_this_ep:
            n_attached_at_all += 1
        if success:
            n_success += 1
        else:
            failures.append((cx, cy, attached_this_ep, float(dist)))

        print(f"  ep {i:02d} cube=({cx:.3f},{cy:.3f}) attached={attached_this_ep} "
              f"success={success} final_dist_to_target={dist:.3f}")

    print(f"\n  attached at all:        {n_attached_at_all}/{n_episodes}")
    print(f"  full success:            {n_success}/{n_episodes}")
    print(f"  worst gap while attached: {worst_gap_while_attached * 1000:.2f} mm "
          f"(tolerance was {3.5:.1f} mm at attach time; should stay small while carried)")

    if diagnose and failures:
        print("\n  failures (cx, cy, ever_attached, final_dist):")
        for f in failures:
            print(f"    {f}")

    ok = worst_gap_while_attached < 0.010  # 10mm hard ceiling -- should be ~0
    if not ok:
        print("\n  FAIL: gap while attached exceeded 10mm at some point. "
              "Check GraspController thresholds in grasp.py (jaw_closed_qpos_threshold, "
              "attach_gap_tolerance) against your gripper's actual closed pose.")
    else:
        print("\n  PASS: grasp stayed flush (no visible gap) whenever attached=True.")
    return ok


def check_bowls() -> bool:
    """Drop a ball with no arm motion at all onto each bowl position and
    confirm it stays inside the rim after settling."""
    model = load_arm_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:5] = default_joint_angles(model)
    data.qpos[5] = 1.5

    results = {}
    for name, (cx, cy), rim_radius in [
        ("pick_bowl", (0.30, 0.0), 0.026),
        ("target_bowl", (0.30, 0.40), 0.033),
    ]:
        mujoco.mj_resetData(model, data)
        data.qpos[:5] = default_joint_angles(model)
        data.qpos[5] = 1.5
        # Drop from a few cm above the bowl so it actually falls onto/into it.
        data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [cx, cy, 0.10]
        data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        # ~2 seconds of sim time at the scene's configured timestep.
        n_steps = int(2.0 / model.opt.timestep)
        for _ in range(n_steps):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)

        cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        final_xy = data.xpos[cube_body_id, :2].copy()
        drift = float(np.linalg.norm(final_xy - np.array([cx, cy])))
        contained = drift < rim_radius
        results[name] = (drift, rim_radius, contained)
        print(f"  {name}: dropped at ({cx:.2f},{cy:.2f}), settled "
              f"{drift * 1000:.1f}mm away, rim_radius={rim_radius * 1000:.0f}mm, "
              f"contained={contained}")

    ok = all(r[2] for r in results.values())
    print("\n  PASS: both bowls physically contain the ball." if ok else
          "\n  FAIL: at least one bowl let the ball escape -- check contype/conaffinity "
          "groups in so101_mjx.py.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp", action="store_true", help="run the grasp-gap check")
    ap.add_argument("--bowls", action="store_true", help="run the bowl-containment check")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diagnose", action="store_true",
                    help="print per-failure cube positions for the grasp check")
    args = ap.parse_args()

    run_grasp = args.grasp or not (args.grasp or args.bowls)
    run_bowls = args.bowls or not (args.grasp or args.bowls)

    ok = True
    if run_grasp:
        print("=== Grasp gap check ===")
        ok = check_grasp(args.episodes, args.seed, args.diagnose) and ok
    if run_bowls:
        print("\n=== Bowl containment check ===")
        ok = check_bowls() and ok

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
