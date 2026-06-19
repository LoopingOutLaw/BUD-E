"""Batch-record scripted pick-and-place episodes into LeRobot v3 layout.

Uses the (now physically-gated, see grasp.py) ScriptedPickAndPlace. The
pick bowl in the world physically constrains the ball during the grasp
phase, so no ball-drift-during-descent videos are recorded.

GRASP FIX: episodes are now only written to the training set if the grasp
genuinely succeeded (ball ended up near the target AND was actually,
physically grasped along the way -- not just "ended up near the target
because the old code teleported it there regardless of jaw alignment").
Previously every episode was written unconditionally, including ones
where the scripted policy missed the ball entirely, polluting the
training data with "the arm waves around empty-handed and the success
metric doesn't catch it" episodes whenever the (now-removed) magic-offset
carry made misses look like hits.

Usage (headless, 100 episodes):
    unset PYTHONPATH
    /home/aditya/.bude-venv/bin/python scripts/record_pick_episodes.py

Usage (live viewer, 5-episode smoke test):
    MUJOCO_GL=glfw DISPLAY=:1 XDG_RUNTIME_DIR=/tmp \\
    /home/aditya/.bude-venv/bin/python scripts/record_pick_episodes.py \\
        --render --max-eps 5 --out /tmp/pick_smoke --seed 42
"""
from __future__ import annotations
import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from bude_vla.data.lerobot_v3 import write_episode, finalize_dataset
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace
from bude_vla.envs.so101_mjx import (
    load_arm_model, default_joint_angles,
    ARM_QPOS_START, ARM_QPOS_END,
    GRIPPER_QPOS_START,
    CUBE_QPOS_START,
)

INSTRUCTION = "pick up the red ball from its bowl and place it in the blue target zone"


def _main_loop(model, data, policy, renderer, cam_ids, max_steps=500):
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")

    images, proprios, actions = [], [], []
    ever_grasped = False

    for step in range(max_steps):
        renderer.update_scene(data, camera=cam_ids[0])
        oh = np.asarray(renderer.render()).copy()
        renderer.update_scene(data, camera=cam_ids[1])
        wr = np.asarray(renderer.render()).copy()
        images.append(np.concatenate([oh, wr], axis=-1).copy())

        proprios.append(data.qpos[ARM_QPOS_START:GRIPPER_QPOS_START + 1].astype(np.float32).copy())

        ctrl, arm_target, done, info = policy.step(model, data)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        if info.get("attached"):
            ever_grasped = True

        action = np.concatenate([
            data.qpos[ARM_QPOS_START:ARM_QPOS_END],
            [data.ctrl[GRIPPER_QPOS_START]],
        ]).astype(np.float32)
        actions.append(action)

        if done:
            break

    cube_final = data.xpos[cube_body_id].copy()
    target_pos = data.xpos[target_body_id].copy()
    reached_target = bool(np.linalg.norm(cube_final[:2] - target_pos[:2]) < 0.05)
    # Honest success now requires BOTH that the ball ended up at the
    # target AND that it got there via a real, physically-gated grasp --
    # not e.g. having been accidentally pushed/rolled there by the arm.
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
    ap.add_argument("--max-eps", type=int, default=100)
    ap.add_argument("--out", default="/home/aditya/bude_vla/data/pick_v4_ball")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--keep-failures", action="store_true",
                    help="Write every episode regardless of outcome "
                         "(old behavior). Off by default -- failed/ungrasped "
                         "episodes are excluded from the training set.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = args.out
    os.makedirs(root, exist_ok=True)

    model = load_arm_model()
    camera_names = ["front_top", "pov"]
    cam_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cn)
                    for cn in camera_names)

    n_success = 0
    n_grasped_but_missed_target = 0
    n_never_grasped = 0
    n_written = 0
    t0 = time.time()

    for i in range(args.max_eps):
        cx = float(rng.uniform(0.28, 0.32))
        cy = float(rng.uniform(-0.02, 0.02))

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[:5] = default_joint_angles(model)
        data.qpos[5] = 1.5
        data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [cx, cy, 0.0295]  # Fix A: ball on bowl floor + radius, not buried
        data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        policy = ScriptedPickAndPlace(model, data,
                                       cube_start_xy=np.array([cx, cy]))
        renderer = mujoco.Renderer(model, height=args.img_size, width=args.img_size)

        ep = _main_loop(model, data, policy, renderer, cam_ids)

        renderer.close()

        if ep["success"]:
            n_success += 1
        elif ep["ever_grasped"]:
            n_grasped_but_missed_target += 1
        else:
            n_never_grasped += 1

        if ep["success"] or args.keep_failures:
            write_episode(root, ep)
            n_written += 1

        print(f"  ep {i:03d}  cube=({cx:.2f},{cy:.2f})  "
              f"steps={len(ep['actions'])}  success={ep['success']}  "
              f"grasped={ep['ever_grasped']}  reached_target={ep['reached_target']}"
              f"{'  [written]' if (ep['success'] or args.keep_failures) else '  [skipped]'}")

    elapsed = time.time() - t0
    rate = n_success / args.max_eps * 100 if args.max_eps > 0 else 0
    print(f"\n=== DONE  {n_success}/{args.max_eps} success ({rate:.0f}%)  "
          f"in {elapsed:.0f}s  out={root} ===")
    print(f"  never grasped the ball at all: {n_never_grasped}")
    print(f"  grasped but missed the target: {n_grasped_but_missed_target}")
    print(f"  episodes written to training set: {n_written}")
    if n_success < args.max_eps * 0.5:
        print(
            "  NOTE: success rate is below 50%. Before assuming the scripted\n"
            "  policy itself needs retuning, run scripts/verify_grasp_fix.py --diagnose\n"
            "  to see whether misses are concentrated at a particular cube (cx, cy) --\n"
            "  that usually means the IK/grasp-pose offsets need a small tweak, not a\n"
            "  fundamental redesign."
        )

    if n_written == 0:
        print("  No episodes were written -- nothing to finalize.")
        return

    stats = finalize_dataset(root)
    print(f"action_normalization persisted to {root}/meta/info.json: "
          f"lo[0:3]={stats['lo'][:3]} ... hi[0:3]={stats['hi'][:3]}")


if __name__ == "__main__":
    main()
