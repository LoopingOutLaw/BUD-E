"""Record a scripted pick-and-place episode as an MP4 video.

Uses the fixed ScriptedPickAndPlace policy (no trained checkpoint needed)
and the portfolio side-front camera so you can see approach, jaw-close,
lift, move, and release in one smooth clip.

Run:
    unset PYTHONPATH
    MUJOCO_GL=egl PYTHONPATH=src python scripts/record_scripted_video.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import imageio
import imageio_ffmpeg
import mujoco
import numpy as np

from bude_vla.envs.so101_mjx import (
    load_arm_model,
    default_joint_angles,
    GRIPPER_QPOS_START,
    CUBE_QPOS_START,
)
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

WIDTH = 640
HEIGHT = 480
FPS = 30
SUBSTEPS_PER_FRAME = 3

# Phase name mapping for debugging
PHASE_NAMES = {
    0: "APPROACH",
    1: "PRE_CLOSE",
    2: "DESCENT",
    3: "GRIP",
    4: "LIFT",
    5: "MOVE",
    6: "RELEASE",
}


def record_one(
    model: mujoco.MjModel,
    cx: float,
    cy: float,
    camera: str = "portfolio",
    verbose: bool = False,
) -> tuple[list[np.ndarray], bool, bool, list[str]]:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:5] = default_joint_angles(model)
    data.qpos[GRIPPER_QPOS_START] = 1.5  # jaw wide open
    data.qpos[CUBE_QPOS_START : CUBE_QPOS_START + 3] = [cx, cy, 0.025]
    data.qpos[CUBE_QPOS_START + 3 : CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array([cx, cy]))
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        cam_id = -1

    frames: list[np.ndarray] = []
    attached_ever = False
    phase_log = []
    last_phase = -1
    lowest_z_recorded = 999.0

    for step in range(900):
        ctrl, arm_target, done, info = policy.step(model, data)
        data.ctrl[:] = ctrl

        for _ in range(SUBSTEPS_PER_FRAME):
            mujoco.mj_step(model, data)

        # Track lowest gripper z
        gripper_bid = policy.gripper_body_id
        mjaw_bid = policy.mjaw_body_id
        for i in range(model.ngeom):
            bid = model.geom_bodyid[i]
            if bid == gripper_bid or bid == mjaw_bid:
                gz = float(data.geom_xpos[i][2])
                gtype = model.geom_type[i]
                if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                    half_z = float(model.geom_size[i][2])
                    lowest_z_recorded = min(lowest_z_recorded, gz - half_z)
                elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
                    rbound = float(model.geom_rbound[i])
                    lowest_z_recorded = min(lowest_z_recorded, gz - rbound)

        # Log phase transitions
        cur_phase = info.get("phase", -1)
        if cur_phase != last_phase:
            pname = PHASE_NAMES.get(cur_phase, f"phase_{cur_phase}")
            phase_log.append(f"step={step}: {pname}")
            last_phase = cur_phase

        if cam_id >= 0:
            renderer.update_scene(data, camera=cam_id)
        else:
            renderer.update_scene(data)
        rgb = renderer.render()
        frames.append(np.asarray(rgb).copy())

        if info.get("attached"):
            attached_ever = True
        if done:
            break

    if verbose:
        for msg in phase_log:
            print(f"    {msg}")
        print(f"    lowest_gripper_z = {lowest_z_recorded:.4f}")
        if lowest_z_recorded < 0:
            print(f"    ⚠ GROUND PENETRATION: {lowest_z_recorded:.4f}")
        else:
            print(f"    ✓ No ground penetration")

    target_xyz = np.array([policy.target_xy[0], policy.target_xy[1], 0.010])
    ball_final = data.xpos[policy.cube_body_id].copy()
    success = float(np.linalg.norm(ball_final[:2] - target_xyz[:2])) < 0.033

    renderer.close()
    return frames, attached_ever, success, phase_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="portfolio")
    ap.add_argument("--attempts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true", help="print phase transitions")
    ap.add_argument(
        "--any-attach",
        action="store_true",
        help="keep the first episode where the jaw attaches, even if the ball is dropped later",
    )
    args = ap.parse_args()

    model = load_arm_model()
    rng = np.random.default_rng(args.seed)

    best_frames = None
    best_tag = ""
    best_priority = -1

    for attempt in range(args.attempts):
        cx = float(rng.uniform(0.285, 0.315))
        cy = float(rng.uniform(-0.015, 0.015))
        frames, attached, success, _ = record_one(
            model, cx, cy, camera=args.camera, verbose=args.verbose,
        )

        if success:
            priority = 2
            tag = f"SUCCESS  cx={cx:.3f} cy={cy:+.3f}"
        elif attached and args.any_attach:
            priority = 1
            tag = f"ATTACHED cx={cx:.3f} cy={cy:+.3f}"
        else:
            priority = 0
            tag = f"miss     cx={cx:.3f} cy={cy:+.3f}"

        label = "✓" if priority > 0 else " "
        print(f"  [{label}] attempt {attempt + 1:2d}  {tag}  frames={len(frames)}")

        if priority > best_priority:
            best_frames = frames
            best_tag = tag
            best_priority = priority

        if priority == 2:
            break

    if best_frames is None:
        print("No frames captured at all.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "..", "demos", "videos")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "grasp_v2_portfolio.mp4")

    with imageio.get_writer(out_path, fps=FPS, macro_block_size=1) as w:
        for f in best_frames:
            w.append_data(f)

    dur = len(best_frames) / FPS
    print(f"\nWrote {out_path}  ({len(best_frames)} frames @ {FPS} fps = {dur:.1f} s)")
    print(f"  episode: {best_tag}")


if __name__ == "__main__":
    main()
