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
from bude_vla.data.lerobot_v3 import write_episode
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace
from bude_vla.envs.so101_mjx import ARM_MODEL_PATH

INSTRUCTION = "pick up the red cube and place it in the blue target zone"


def _main_loop(model, data, policy, renderer, cam_id, viewer=None,
               max_steps=350, step_delay=0.0, use_free_cam=True):
    images, proprios, actions = [], [], []
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                       "target_zone")

    for step in range(max_steps):
        if use_free_cam:
            renderer.update_scene(data)
        else:
            renderer.update_scene(data, camera=cam_id)
        images.append(np.asarray(renderer.render()).copy())
        arm_proprio = data.qpos[7:15].astype(np.float32).copy()
        cube_xyz = data.xpos[cube_body_id].astype(np.float32).copy()
        proprios.append(
            np.concatenate([arm_proprio, cube_xyz]).astype(np.float32)
        )

        ctrl, arm_target, done, _ = policy.step(model, data)
        kinematic_action = np.concatenate([arm_target, [ctrl[6]]]).astype(np.float32)
        actions.append(kinematic_action)

        data.ctrl[:] = 0.0
        data.ctrl[6] = ctrl[6]
        data.qvel[6:12] = 0.0
        data.qpos[7:13] = arm_target
        policy._carry_cube_with(data)
        mujoco.mj_step(model, data)
        data.qpos[7:13] = arm_target
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
    ap.add_argument("--cam", default=None,
                    help="Camera name (e.g. front_top). Default: free camera "
                         "(auto-framing, same as push_v3)")
    ap.add_argument("--img-size", type=int, default=64,
                    help="Render resolution (default 64; use 224 for VLA training)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = args.out
    os.makedirs(root, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
    use_free_cam = args.cam is None
    cam_id = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
              if args.cam else -1)

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

        if args.render:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                ep = _main_loop(model, data, policy, renderer, cam_id,
                                viewer=viewer, step_delay=args.slow,
                                use_free_cam=use_free_cam)
                del viewer
        else:
            ep = _main_loop(model, data, policy, renderer, cam_id,
                            viewer=None, step_delay=args.slow,
                            use_free_cam=use_free_cam)

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


if __name__ == "__main__":
    main()
