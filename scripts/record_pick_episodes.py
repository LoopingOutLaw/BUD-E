"""Batch-record scripted pick-and-place episodes into LeRobot v3 layout.

Uses the v7 ggand0/pick-101 approach — physics-only grasping, no kinematic
carry/teleport. Only writes successful episodes to the training set.

Usage (headless, 100 episodes):
    unset PYTHONPATH
    MUJOCO_GL=egl PYTHONPATH=src python scripts/record_pick_episodes.py

Usage (smoke test, 5 episodes):
    MUJOCO_GL=egl PYTHONPATH=src python scripts/record_pick_episodes.py --max-eps 5
"""
from __future__ import annotations
import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from bude_vla.data.lerobot_v3 import write_episode, finalize_dataset
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace, PHASE_NAMES
from bude_vla.envs.so101_mjx import (
    load_arm_model,
    ARM_QPOS_START, ARM_QPOS_END,
    GRIPPER_QPOS_START,
    CUBE_QPOS_START,
    CUBE_REST_Z,
    is_grasping_from_contacts,
)

INSTRUCTION = "pick up the red cube and place it in the blue target zone"

SUBSTEPS_PER_FRAME = 4  # match video recorder


def _main_loop(model, data, policy, renderer, cam_ids, max_steps=2000):
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    gripperframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    images, proprios, actions = [], [], []
    ever_grasped = False

    for step in range(max_steps):
        # Dual-cam render (overhead + wrist)
        renderer.update_scene(data, camera=cam_ids[0])
        oh = np.asarray(renderer.render()).copy()
        renderer.update_scene(data, camera=cam_ids[1])
        wr = np.asarray(renderer.render()).copy()
        images.append(np.concatenate([oh, wr], axis=-1).copy())

        # Check if currently grasping (shared contact helper — matches eval exactly)
        is_grasping = is_grasping_from_contacts(model, data)
        if is_grasping:
            ever_grasped = True

        # 9D proprio: arm(5) + gripper(1) + target_rel(2) + is_grasping(1)
        gripper_pos = data.site_xpos[gripperframe_id]
        target_pos = data.xpos[target_body_id]
        target_rel = target_pos[:2] - gripper_pos[:2]

        proprio = np.concatenate([
            data.qpos[ARM_QPOS_START:GRIPPER_QPOS_START + 1],  # 6D
            target_rel,                                          # 2D
            [is_grasping],                                       # 1D = 9D total
        ]).astype(np.float32)
        proprios.append(proprio)

        # Policy step — returns ctrl, recorder calls mj_step
        ctrl, arm_target, done, info = policy.step(model, data)
        data.ctrl[:] = ctrl
        for _ in range(SUBSTEPS_PER_FRAME):
            mujoco.mj_step(model, data)

        if info.get("attached"):
            ever_grasped = True

        # Action = target arm ctrl + gripper ctrl (what the policy commanded)
        action = np.concatenate([
            ctrl[ARM_QPOS_START:ARM_QPOS_END],
            [ctrl[GRIPPER_QPOS_START]],
        ]).astype(np.float32)
        actions.append(action)

        if done:
            break

    cube_final = data.xpos[cube_body_id].copy()
    target_pos = data.xpos[target_body_id].copy()
    reached_target = bool(np.linalg.norm(cube_final[:2] - target_pos[:2]) < 0.05)
    success = bool(reached_target and ever_grasped)

    return {
        "images": np.array(images, dtype=np.uint8),
        "proprio": np.array(proprios, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "instruction": INSTRUCTION,
        "success": success,
        "ever_grasped": ever_grasped,
        "reached_target": reached_target,
        "cube_final_xyz": cube_final,
        "target_xyz": target_pos,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eps", type=int, default=500)
    ap.add_argument("--out", default="/home/aditya/bude_vla/data/pick_v10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--recovery-jitter-xy", type=float, default=0.0,
                    help="Max XY waypoint jitter in meters during approach/descent, decayed to zero before close.")
    ap.add_argument("--recovery-jitter-z", type=float, default=0.0,
                    help="Max Z/depth jitter in meters during descent, decayed to zero before close.")
    ap.add_argument("--recovery-jitter-prob", type=float, default=0.0,
                    help="Probability that an episode uses recoverable approach/descent jitter.")
    ap.add_argument("--max-grasp-retries", type=int, default=0,
                    help="Number of failed close attempts the scripted expert may recover from.")
    ap.add_argument("--nudge-recovery-prob", type=float, default=0.0,
                    help="Probability of a light touch/nudge during descent followed by backoff and clean retry.")
    ap.add_argument("--nudge-recovery-xy", type=float, default=0.0,
                    help="Max XY offset in meters for the induced descent nudge before recovery.")
    ap.add_argument("--nudge-recovery-z", type=float, default=0.0,
                    help="Max downward Z offset in meters for the induced descent nudge before recovery.")
    ap.add_argument("--retry-miss-xy", type=float, default=0.0,
                    help="Max XY close-target miss in meters on the first attempt for retry demos.")
    ap.add_argument("--retry-miss-prob", type=float, default=0.0,
                    help="Probability that an episode starts with an induced first-attempt miss.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = args.out
    os.makedirs(root, exist_ok=True)

    model = load_arm_model()
    camera_names = ["front_top", "wrist_cam"]
    cam_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cn)
                    for cn in camera_names)

    n_success = 0
    n_grasped_but_missed = 0
    n_never_grasped = 0
    n_written = 0
    t0 = time.time()

    if (args.recovery_jitter_xy > 0.0 or args.recovery_jitter_z > 0.0) and args.recovery_jitter_prob > 0.0:
        print("  recovery jitter enabled: "
              f"xy=+/-{args.recovery_jitter_xy:.3f}m  "
              f"z=+/-{args.recovery_jitter_z:.3f}m  "
              f"prob={args.recovery_jitter_prob:.2f}")
    if args.max_grasp_retries > 0:
        print("  grasp retry demos enabled: "
              f"max_retries={args.max_grasp_retries}  "
              f"miss=+/-{args.retry_miss_xy:.3f}m  prob={args.retry_miss_prob:.2f}")
    if args.nudge_recovery_prob > 0.0:
        print("  nudge recovery demos enabled: "
              f"xy=+/-{args.nudge_recovery_xy:.3f}m  "
              f"z=-{args.nudge_recovery_z:.3f}m  "
              f"prob={args.nudge_recovery_prob:.2f}")

    for i in range(args.max_eps):
        # Cube position: wide randomization to force visual grounding
        cx = float(rng.uniform(0.15, 0.35))
        cy = float(rng.uniform(-0.10, 0.10))

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Top-down arm configuration (matching pick-101)
        data.qpos[0] = 0.0    # shoulder_pan
        data.qpos[1] = -0.5   # shoulder_lift
        data.qpos[2] = 0.95   # elbow_flex
        data.qpos[3] = np.pi / 2  # wrist_flex (pointing down)
        data.qpos[4] = np.pi / 2  # wrist_roll (fingers along Y)
        data.qpos[GRIPPER_QPOS_START] = 0.3  # partially open

        # Cube on floor (3cm cube, z=0.015)
        data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [cx, cy, CUBE_REST_Z]
        data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]

        # Set initial ctrl to match arm config
        data.ctrl[:5] = data.qpos[:5]
        data.ctrl[GRIPPER_QPOS_START] = 0.3
        mujoco.mj_forward(model, data)

        # Let cube settle
        for _ in range(50):
            mujoco.mj_step(model, data)

        policy = ScriptedPickAndPlace(
            model,
            data,
            cube_start_xy=np.array([cx, cy]),
            recovery_jitter_xy=args.recovery_jitter_xy,
            recovery_jitter_z=args.recovery_jitter_z,
            recovery_jitter_prob=args.recovery_jitter_prob,
            max_grasp_retries=args.max_grasp_retries,
            nudge_recovery_prob=args.nudge_recovery_prob,
            nudge_recovery_xy=args.nudge_recovery_xy,
            nudge_recovery_z=args.nudge_recovery_z,
            retry_miss_xy=args.retry_miss_xy,
            retry_miss_prob=args.retry_miss_prob,
            rng=rng,
        )
        renderer = mujoco.Renderer(model, height=args.img_size, width=args.img_size)

        ep = _main_loop(model, data, policy, renderer, cam_ids)

        renderer.close()

        if ep["success"]:
            n_success += 1
        elif ep["ever_grasped"]:
            n_grasped_but_missed += 1
        else:
            n_never_grasped += 1

        if ep["success"] or args.keep_failures:
            write_episode(root, ep)
            n_written += 1

        print(f"  ep {i:03d}  cube=({cx:.3f},{cy:.3f})  "
              f"steps={len(ep['actions'])}  success={ep['success']}  "
              f"grasped={ep['ever_grasped']}  reached={ep['reached_target']}"
              f"{'  [written]' if (ep['success'] or args.keep_failures) else '  [skipped]'}")

    elapsed = time.time() - t0
    rate = n_success / args.max_eps * 100 if args.max_eps > 0 else 0
    print(f"\n=== DONE  {n_success}/{args.max_eps} success ({rate:.0f}%)  "
          f"in {elapsed:.0f}s  out={root} ===")
    print(f"  never grasped: {n_never_grasped}")
    print(f"  grasped but missed target: {n_grasped_but_missed}")
    print(f"  episodes written: {n_written}")

    if n_written == 0:
        print("  No episodes written — nothing to finalize.")
        return

    stats = finalize_dataset(root)
    print(f"action_normalization persisted: "
          f"lo[0:3]={stats['lo'][:3]} ... hi[0:3]={stats['hi'][:3]}")


if __name__ == "__main__":
    main()
