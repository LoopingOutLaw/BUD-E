"""Roll out the scripted pick-and-place policy in CPU MuJoCo and write an MP4.

This is the "demo" video — high-res, scripted baseline, deterministic. Used
as the portfolio deliverable until the trained network drives a longer /
varied sequence.

Usage:
    PYTHONPATH=src python scripts/render_pick_rollout.py
    PYTHONPATH=src python scripts/render_pick_rollout.py \
        --out demos/videos/pick_rollout.mp4 \
        --num-rollouts 5 --seed 42
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bude_vla.envs.so101_mjx import ARM_MODEL_PATH
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace


def _font(size: int = 18):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


def _overlay_line(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil)
    font = _font(20)
    w, h = pil.size
    pad = 8
    box_h = font.size + pad * 2
    draw.rectangle([(0, 0), (w, box_h)], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 220, 0), font=font)
    return np.asarray(pil)


def _rollout(model, cx: float, cy: float, img_size: int,
             max_steps: int, use_free_cam: bool = True,
             cam_name: str = "portfolio", frame_repeat: int = 1) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [cx, cy, 0.445]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    policy = ScriptedPickAndPlace(model, data,
                                   cube_start_xy=np.array([cx, cy]))
    renderer = mujoco.Renderer(model, height=img_size, width=img_size)
    cam_id = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
              if not use_free_cam else -1)

    frames = []
    done = False
    for _ in range(max_steps):
        if use_free_cam:
            renderer.update_scene(data)
        else:
            renderer.update_scene(data, camera=cam_id)
        img = np.asarray(renderer.render()).copy()
        for _ in range(frame_repeat):
            frames.append(img)

        ctrl, arm_target, done, _ = policy.step(model, data)
        data.ctrl[:] = 0.0
        data.ctrl[6] = ctrl[6]
        data.qvel[6:12] = 0.0
        data.qpos[7:13] = arm_target
        policy._carry_cube_with(data)
        mujoco.mj_step(model, data)
        data.qpos[7:13] = arm_target
        policy._carry_cube_with(data)
        if done:
            break
    renderer.close()

    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    cube_final = data.xpos[cube_body].copy()
    target_pos = data.xpos[target_body].copy()
    success = bool(np.linalg.norm(cube_final[:2] - target_pos[:2]) < 0.10)
    return {"frames": frames, "success": success, "cube_final": cube_final,
            "target_pos": target_pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(ARM_MODEL_PATH))
    ap.add_argument("--out", default="/home/aditya/bude_vla/demos/videos/"
                                       "pick_rollout.mp4")
    ap.add_argument("--num-rollouts", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--max-steps", type=int, default=350)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cam", default="portfolio",
                    help="Camera name when not using free cam")
    ap.add_argument("--no-free-cam", action="store_true",
                    help="Use named XML camera instead of auto-framing free cam")
    ap.add_argument("--slow", type=int, default=6,
                    help="Repeat each sim frame N times in the video. "
                         "6=~real-time, 12=half-speed (default: 6)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model_path)
    rng = np.random.default_rng(args.seed)

    all_frames = []
    successes = 0
    for k in range(args.num_rollouts):
        cx = float(rng.uniform(0.50, 0.75))
        cy = float(rng.uniform(-0.15, 0.15))
        out = _rollout(model, cx, cy, args.img_size, args.max_steps,
                       use_free_cam=not args.no_free_cam,
                       cam_name=args.cam, frame_repeat=args.slow)
        successes += int(out["success"])
        labeled = [_overlay_line(f,
                                 f"Pick #{k+1}/{args.num_rollouts}  cube="
                                 f"({cx:+.2f},{cy:+.2f})  "
                                 f"success={out['success']}")
                   for f in out["frames"]]
        all_frames.extend(labeled)
        print(f"  rollout {k+1}: cube=({cx:+.2f},{cy:+.2f}) "
              f"-> target=({out['target_pos'][0]:.2f},{out['target_pos'][1]:.2f}) "
              f"success={out['success']} frames={len(labeled)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), all_frames, fps=30, codec="libx264",
                    quality=8)
    rate = successes / max(1, args.num_rollouts) * 100
    print(f"\n=== DONE  {successes}/{args.num_rollouts} success ({rate:.0f}%)  "
          f"frames={len(all_frames)}  ->  {out_path} ===")


if __name__ == "__main__":
    main()
