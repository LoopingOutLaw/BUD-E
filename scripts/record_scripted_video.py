"""Record a scripted pick-and-place episode as an MP4 video.

Uses ggand0/pick-101 approach. Policy sets data.ctrl — recorder calls mj_step.

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
    GRIPPER_QPOS_START,
    CUBE_QPOS_START,
    CUBE_REST_Z,
)
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace, PHASE_NAMES

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

WIDTH = 640
HEIGHT = 480
FPS = 30
SUBSTEPS_PER_FRAME = 4


def record_one(
    model: mujoco.MjModel,
    cx: float,
    cy: float,
    camera: str = "portfolio",
    verbose: bool = False,
) -> tuple[list[np.ndarray], bool, bool, list[str]]:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Top-down arm configuration (matching pick-101)
    data.qpos[0] = 0.0    # shoulder_pan
    data.qpos[1] = -0.5   # shoulder_lift
    data.qpos[2] = 0.95   # elbow_flex
    data.qpos[3] = np.pi / 2  # wrist_flex (pointing down)
    data.qpos[4] = np.pi / 2  # wrist_roll (fingers spread along Y)
    data.qpos[GRIPPER_QPOS_START] = 0.3  # partially open

    # Cube on floor (matching pick-101: 3cm cube at z=0.015)
    data.qpos[CUBE_QPOS_START : CUBE_QPOS_START + 3] = [cx, cy, CUBE_REST_Z]
    data.qpos[CUBE_QPOS_START + 3 : CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]

    # Set initial ctrl to match the arm config
    data.ctrl[:5] = data.qpos[:5]
    data.ctrl[GRIPPER_QPOS_START] = 0.3
    mujoco.mj_forward(model, data)

    # Let cube settle (matching pick-101)
    for _ in range(50):
        mujoco.mj_step(model, data)

    policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array([cx, cy]))
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        cam_id = -1

    frames: list[np.ndarray] = []
    attached_ever = False
    phase_log = []
    last_phase = -1

    for step in range(2000):
        ctrl, arm_target, done, info = policy.step(model, data)
        data.ctrl[:] = ctrl

        for _ in range(SUBSTEPS_PER_FRAME):
            mujoco.mj_step(model, data)

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
        cube_final = data.xpos[policy.cube_body_id].copy()
        print(f"    cube_final = [{cube_final[0]:.4f}, {cube_final[1]:.4f}, {cube_final[2]:.4f}]")
        print(f"    grasping = {info.get('grasping', False)}")

    target_xyz = np.array([policy.target_xy[0], policy.target_xy[1], CUBE_REST_Z])
    cube_final = data.xpos[policy.cube_body_id].copy()
    success = attached_ever and float(np.linalg.norm(cube_final[:2] - target_xyz[:2])) < 0.033

    renderer.close()
    return frames, attached_ever, success, phase_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="portfolio")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model = load_arm_model()

    # Cube position matching pick-101
    cx, cy = 0.25, 0.0

    best_frames = None
    best_tag = ""
    best_priority = -1

    for attempt in range(args.attempts):
        frames, attached, success, _ = record_one(
            model, cx, cy, camera=args.camera, verbose=args.verbose,
        )

        if success:
            priority = 2
            tag = "SUCCESS"
        elif attached:
            priority = 1
            tag = "ATTACHED"
        else:
            priority = 0
            tag = "miss"

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
    out_path = os.path.join(out_dir, "grasp_v7_portfolio.mp4")

    with imageio.get_writer(out_path, fps=FPS, macro_block_size=1) as w:
        for f in best_frames:
            w.append_data(f)

    dur = len(best_frames) / FPS
    print(f"\nWrote {out_path}  ({len(best_frames)} frames @ {FPS} fps = {dur:.1f} s)")
    print(f"  episode: {best_tag}")


if __name__ == "__main__":
    main()
