"""Batch-record scripted pick-and-place episodes into LeRobot v3 layout.

Usage (headless, 100 episodes):
    unset PYTHONPATH
    PYTHONPATH=../src python scripts/record_pick_episodes.py

Usage (live viewer, 5-episode smoke test):
    MUJOCO_GL=glfw DISPLAY=:1 XDG_RUNTIME_DIR=/tmp \
    PYTHONPATH=../src python scripts/record_pick_episodes.py \
        --render --max-eps 5 --out /tmp/pick_smoke --seed 42
"""
from __future__ import annotations
import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import mujoco.viewer
from bude_vla.data.lerobot_v3 import write_episode, finalize_dataset
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace
from bude_vla.envs.so101_mjx import ARM_MODEL_PATH

INSTRUCTION = "pick up the red cube and place it in the blue target zone"


def _main_loop(model, data, policy, renderer, cam_ids, viewer=None,
               max_steps=350, step_delay=0.0):
    SMOOTH_STEPS = 10
    SMOOTH_FRAC = 0.25

    images, proprios, actions = [], [], []
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                       "target_zone")

    def _render_dual():
        renderer.update_scene(data, camera=cam_ids[0])
        oh = np.asarray(renderer.render()).copy()
        renderer.update_scene(data, camera=cam_ids[1])
        wr = np.asarray(renderer.render()).copy()
        return np.concatenate([oh, wr], axis=-1).copy()

    for step in range(max_steps):
        images.append(_render_dual())
        proprios.append(data.qpos[7:15].astype(np.float32).copy())

        ctrl, arm_target, done, _ = policy.step(model, data)
        kinematic_action = np.concatenate([arm_target, [ctrl[6]]]).astype(np.float32)
        actions.append(kinematic_action)

        tgt = np.clip(arm_target, -3.5, 3.5).astype(np.float64)
        cur = data.qpos[7:13].astype(np.float64).copy()
        for k in range(SMOOTH_STEPS):
            err = tgt - cur
            cur = cur + err * SMOOTH_FRAC
            data.ctrl[:] = 0.0
            data.ctrl[6] = ctrl[6]
            data.qvel[6:12] = 0.0
            data.qpos[7:13] = cur
            policy._carry_cube_with(data)
            mujoco.mj_step(model, data)
        data.qpos[7:13] = tgt
        policy._carry_cube_with(data)

        if viewer is not None and step % 4 == 0:
            viewer.sync()
        if step_delay > 0:
            time.sleep(step_delay)

        if done:
            break

    if viewer is not None:
        hold_steps = max(30, int(1.5 / max(step_delay, 1e-3)))
        for _ in range(hold_steps):
            viewer.sync()
            if step_delay > 0:
                time.sleep(step_delay)

    cube_final = data.xpos[cube_body_id].copy()
    target_pos = data.xpos[target_body_id].copy()
    success = bool(np.linalg.norm(cube_final[:2] - target_pos[:2]) < 0.10)

    return {
        "images": np.array(images, dtype=np.uint8),
        "proprio": np.array(proprios, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "instruction": INSTRUCTION,
        "success": success,
        "cube_final_xyz": cube_final,
        "target_xyz": target_pos,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eps", type=int, default=100)
    ap.add_argument("--out", default="/home/aditya/bude_vla/data/pick_v3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true",
                    help="show live MuJoCo viewer per episode (needs DISPLAY)")
    ap.add_argument("--slow", type=float, default=0.0,
                    help="seconds to sleep per step (~0.04 = real-time)")
    ap.add_argument("--img-size", type=int, default=64,
                    help="Render resolution (default 64; use 224 for VLA training)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = args.out
    os.makedirs(root, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
    overhead_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                        "front_top")
    gripper_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                       "gripper_cam")
    cam_ids = (overhead_cam_id, gripper_cam_id)

    n_success = 0
    t0 = time.time()

    for i in range(args.max_eps):
        cx = float(rng.uniform(0.50, 0.75))
        cy = float(rng.uniform(-0.15, 0.15))

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [cx, cy, 0.445]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        policy = ScriptedPickAndPlace(model, data,
                                      cube_start_xy=np.array([cx, cy]))
        renderer = mujoco.Renderer(model, height=args.img_size, width=args.img_size)

        ep = _main_loop(model, data, policy, renderer, cam_ids,
                        viewer=None, step_delay=args.slow)

        renderer.close()
        write_episode(root, ep)
        if ep["success"]:
            n_success += 1
        print(f"  ep {i:03d}  cube=({cx:.2f},{cy:.2f})  "
              f"steps={len(ep['actions'])}  success={ep['success']}")

    elapsed = time.time() - t0
    rate = n_success / args.max_eps * 100 if args.max_eps > 0 else 0
    print(f"\n=== DONE  {n_success}/{args.max_eps} success ({rate:.0f}%)  "
          f"in {elapsed:.0f}s  out={root} ===")

    stats = finalize_dataset(root)
    print(f"action_normalization persisted to {root}/meta/info.json: "
          f"lo[0:3]={stats['lo'][:3]} ... hi[0:3]={stats['hi'][:3]}")


if __name__ == "__main__":
    main()
