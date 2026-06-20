#!/usr/bin/env python
"""Capture a single scripted pick-and-place episode to MP4 for visual review.

Unlike record_scripted_video.py, this:
  - Saves EVERY attempt, not just the best one (we need to see what goes wrong).
  - Writes to a timestamped file under /tmp so we don't overwrite the
    shipped demo at demos/videos/grasp_fix_portfolio.mp4.
  - Has a --seed argument for reproducibility.

Run:
    unset PYTHONPATH && /home/aditya/venv-bude/bin/python scripts/record_debug_video.py [--attempts N] [--seed S] [--out PATH]
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import imageio
import imageio_ffmpeg
import mujoco
import numpy as np

from bude_vla.envs.so101_mjx import (
    load_arm_model,
    default_joint_angles,
    CUBE_QPOS_START,
)
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

WIDTH = 640
HEIGHT = 480
FPS = 30
SUBSTEPS_PER_FRAME = 3


def record_one(model, cx, cy, camera="portfolio"):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:5] = default_joint_angles(model)
    data.qpos[5] = 1.5
    data.qpos[CUBE_QPOS_START : CUBE_QPOS_START + 3] = [cx, cy, 0.025]  # cube half-extent on world floor
    data.qpos[CUBE_QPOS_START + 3 : CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array([cx, cy]))
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    cam_id = cam_id if cam_id >= 0 else None

    frames = []
    attached_step = None
    contact_log = []  # (step, contact_pairs)

    for step in range(600):
        ctrl, _, done, info = policy.step(model, data)
        data.ctrl[:] = ctrl
        for _ in range(SUBSTEPS_PER_FRAME):
            mujoco.mj_step(model, data)

        if info.get("attached") and attached_step is None:
            attached_step = step

        # log every 30 steps whether the moving_jaw contacts anything
        if step % 30 == 0:
            contacts = []
            for i in range(data.ncon):
                c = data.contact[i]
                gb1 = model.geom_bodyid[c.geom1]
                gb2 = model.geom_bodyid[c.geom2]
                bname1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, gb1)
                bname2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, gb2)
                if ("jaw" in (bname1 or "") or "jaw" in (bname2 or "")
                    or "gripper" in (bname1 or "") or "gripper" in (bname2 or "")):
                    contacts.append((bname1, bname2))
            contact_log.append((step, list(contacts)))

        if cam_id is not None:
            renderer.update_scene(data, camera=cam_id)
        else:
            renderer.update_scene(data)
        rgb = renderer.render()
        frames.append(np.asarray(rgb).copy())

        if done:
            break

    target_xyz = np.array([policy.target_xy[0], policy.target_xy[1], 0.025])
    ball_final = data.xpos[policy.cube_body_id].copy()
    success = float(np.linalg.norm(ball_final[:2] - target_xyz[:2])) < 0.033

    renderer.close()
    return frames, attached_step, success, contact_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="portfolio")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str,
                    default="/tmp/bude_debug/baseline_episode.mp4")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_arm_model()
    rng = np.random.default_rng(args.seed)

    summary_lines = []
    for attempt in range(args.attempts):
        cx = float(rng.uniform(0.285, 0.315))
        cy = float(rng.uniform(-0.015, 0.015))
        frames, attached_step, success, contacts = record_one(
            model, cx, cy, camera=args.camera
        )
        label = "SUCCESS" if success else ("ATTACHED" if attached_step is not None else "MISS")
        summary_lines.append(
            f"attempt {attempt+1}: {label}  cx={cx:.3f} cy={cy:+.3f} "
            f"attached_step={attached_step} frames={len(frames)} "
            f"jaw_contact_event_count={sum(1 for _, cs in contacts if cs)}"
        )
        for step, cs in contacts:
            if cs:
                summary_lines.append(f"    step {step}: {cs}")
        with imageio.get_writer(str(out_path), fps=FPS, macro_block_size=1) as w:
            for f in frames:
                w.append_data(f)
        summary_lines.append(f"Wrote {out_path}  ({len(frames)} frames @ {FPS} fps)")

    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
