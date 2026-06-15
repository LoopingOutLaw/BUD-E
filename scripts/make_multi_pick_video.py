"""Multi-arm pick-and-place training montage — high-resolution live render.

Runs 12 scripted arms sequentially (shared renderer), composites tiles into
a grid video. Each arm does multiple pick-and-place rounds for a ~10-15s
portfolio video at proper resolution.

Usage:
    PYTHONPATH=src python scripts/make_multi_pick_video.py
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


def _font(size: int = 14):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-arms", type=int, default=12)
    ap.add_argument("--n-rounds", type=int, default=5,
                    help="Each arm does this many pick-and-place rounds")
    ap.add_argument("--cell", type=int, default=160,
                    help="Pixel size per arm tile")
    ap.add_argument("--out", default="/home/aditya/bude_vla/demos/videos/"
                                       "multi_pick_training.mp4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.n_arms
    cell = args.cell
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    pad = 2
    title_h = 38

    grid_w = cols * cell + (cols + 1) * pad
    grid_h = rows * cell + (rows + 1) * pad + title_h

    model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
    renderer = mujoco.Renderer(model, height=cell, width=cell)

    font_title = _font(max(18, cell // 7))
    font_sub = _font(max(13, cell // 10))

    # pre-generate all cube positions
    all_cx = rng.uniform(0.50, 0.75, size=(n, args.n_rounds))
    all_cy = rng.uniform(-0.15, 0.15, size=(n, args.n_rounds))

    # init arm states
    arms = []
    for i in range(n):
        d = mujoco.MjData(model)
        arms.append({"data": d, "policy": None, "done": True,
                      "round": 0})

    frames = []

    def _reset_arm(i, rnd):
        cx = float(all_cx[i, rnd])
        cy = float(all_cy[i, rnd])
        mujoco.mj_resetData(model, arms[i]["data"])
        arms[i]["data"].qpos[0:3] = [cx, cy, 0.445]
        arms[i]["data"].qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, arms[i]["data"])
        arms[i]["policy"] = ScriptedPickAndPlace(
            model, arms[i]["data"], cube_start_xy=np.array([cx, cy]))
        arms[i]["done"] = False
        arms[i]["round"] = rnd

    # start round 0
    for i in range(n):
        _reset_arm(i, 0)

    # stagger arms: arm i starts at step (i * stagger) so they
    # desync and the grid shows varied phases
    stagger = 5
    arm_start_step = [i * stagger for i in range(n)]

    total_rounds = args.n_rounds
    all_finished = False
    global_step = 0
    max_global_steps = (50 + stagger * n) * total_rounds + 60

    for step in range(max_global_steps):
        for i, arm in enumerate(arms):
            if arm["done"]:
                continue
            if global_step < arm_start_step[i]:
                continue
        n_done = 0
        for arm in arms:
            if arm["done"]:
                n_done += 1
                continue
            ctrl, arm_target, done, _ = arm["policy"].step(model, arm["data"])
            arm["data"].ctrl[:] = 0.0
            arm["data"].ctrl[6] = ctrl[6]
            arm["data"].qvel[6:12] = 0.0
            arm["data"].qpos[7:13] = arm_target
            arm["policy"]._carry_cube_with(arm["data"])
            mujoco.mj_step(model, arm["data"])
            arm["data"].qpos[7:13] = arm_target
            arm["policy"]._carry_cube_with(arm["data"])
            arm["done"] = done

        # build grid frame
        canvas = np.full((grid_h, grid_w, 3), 20, dtype=np.uint8)
        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        draw.text((pad + 4, pad),
                  "BUD-E  -  Multi-Arm Pick-and-Place Training",
                  fill=(255, 220, 0), font=font_title)
        canvas = np.asarray(pil).copy()

        for i, arm in enumerate(arms):
            r_i, c_i = divmod(i, cols)
            y0 = title_h + r_i * (cell + pad) + pad
            x0 = c_i * (cell + pad) + pad
            renderer.update_scene(arm["data"])
            tile = np.asarray(renderer.render()).copy()
            canvas[y0:y0 + cell, x0:x0 + cell] = tile

        frames.append(canvas)

        # if all done, advance round
        n_done = sum(1 for a in arms if a["done"])
        if n_done == n:
            cur_round = arms[0]["round"]
            if cur_round + 1 >= total_rounds:
                for _ in range(10):
                    frames.append(canvas.copy())
                break
            for i in range(n):
                _reset_arm(i, cur_round + 1)
                arm_start_step[i] = global_step + i * stagger

        global_step += 1

    renderer.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=args.fps, codec="libx264",
                    quality=8, macro_block_size=1)
    dur = len(frames) / args.fps
    print(f"\n=== DONE  {len(frames)} frames  {dur:.1f}s  "
          f"{n} arms x {total_rounds} rounds  ->  {out_path} ===")


if __name__ == "__main__":
    main()
