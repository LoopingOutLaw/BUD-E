"""One-off per-step trace harness.

Saves a debug video AND prints a per-step line for the moment each
attached-flip happens, including which release_reason path fired.
This is the diagnostic the user asked for after the explicit-release
refactor: tells you whether ball gets dropped because the scripted
RELEASE phase force-released it (correct), or whether one of the
catastrophe-only fallbacks fired during LIFT/MOVE (real bug).

Run:
    unset PYTHONPATH && MUJOCO_GL=egl PYTHONPATH=src \\
        /home/aditya/venv-bude/bin/python scripts/trace_release_reasons.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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

WIDTH, HEIGHT, FPS = 640, 480, 30
SUBSTEPS_PER_FRAME = 3


def trace_one(model, cx, cy, camera="portfolio"):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:5] = default_joint_angles(model)
    data.qpos[5] = 1.5
    data.qpos[CUBE_QPOS_START : CUBE_QPOS_START + 3] = [cx, cy, 0.0295]
    data.qpos[CUBE_QPOS_START + 3 : CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array([cx, cy]))
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    cam_id = cam_id if cam_id >= 0 else None

    frames = []
    prev_attached = False
    release_events = []  # (step, phase, phase_step, reason)
    trace_lines = []

    for step in range(600):
        ctrl, _, done, info = policy.step(model, data)
        data.ctrl[:] = ctrl
        for _ in range(SUBSTEPS_PER_FRAME):
            mujoco.mj_step(model, data)

        attached = bool(info.get("attached"))
        phase = info.get("phase")
        phase_step = info.get("phase_step")
        reason = info.get("release_reason")
        phase_name = {0: "APPROACH", 1: "GRASP", 2: "LIFT", 3: "MOVE", 4: "RELEASE"}.get(phase, str(phase))

        # Print first attached + every attach-flip.
        if attached and not prev_attached:
            jaw_qpos = float(data.qpos[GRIPPER_QPOS_START])
            trace_lines.append(
                f"  step {step:3d} ATTACH  phase={phase_name:8s} phase_step={phase_step:3d} jaw_q={jaw_qpos:+.3f}"
            )
        if not attached and prev_attached:
            jaw_qpos = float(data.qpos[GRIPPER_QPOS_START])
            trace_lines.append(
                f"  step {step:3d} RELEASE phase={phase_name:8s} phase_step={phase_step:3d} "
                f"jaw_q={jaw_qpos:+.3f} reason={reason!r}"
            )
            release_events.append((step, phase, phase_step, reason))
        prev_attached = attached

        # Every 30 steps: where is the ball vs target?
        if step % 30 == 0:
            cube = data.xpos[policy.cube_body_id].copy()
            tgt = np.array([policy.target_xy[0], policy.target_xy[1], 0.0295])
            dist = float(np.linalg.norm(cube[:2] - tgt[:2]))
            trace_lines.append(
                f"  step {step:3d} phase={phase_name:8s} phase_step={phase_step:3d} "
                f"ball=({cube[0]:+.3f},{cube[1]:+.3f},{cube[2]:+.3f}) "
                f"dist_to_target={dist:.3f}m attached={attached}"
            )

        if cam_id is not None:
            renderer.update_scene(data, camera=cam_id)
        else:
            renderer.update_scene(data)
        rgb = renderer.render()
        frames.append(np.asarray(rgb).copy())

        if done:
            break

    target_xyz = np.array([policy.target_xy[0], policy.target_xy[1], 0.0295])
    ball_final = data.xpos[policy.cube_body_id].copy()
    success = float(np.linalg.norm(ball_final[:2] - target_xyz[:2])) < 0.033
    renderer.close()
    return frames, success, trace_lines, release_events


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="portfolio")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="/tmp/bude_debug/v12_explicit_release.mp4")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_arm_model()
    rng = np.random.default_rng(args.seed)

    all_releases = []
    for attempt in range(args.attempts):
        cx = float(rng.uniform(0.285, 0.315))
        cy = float(rng.uniform(-0.015, 0.015))
        print(f"\n=== attempt {attempt+1}  cube=({cx:.3f},{cy:+.3f}) ===")
        frames, success, lines, releases = trace_one(model, cx, cy, camera=args.camera)
        for l in lines:
            print(l)
        print(f"  SUCCESS={success}  release_events={releases}")
        all_releases.extend(releases)
        with imageio.get_writer(str(out_path), fps=FPS, macro_block_size=1) as w:
            for f in frames:
                w.append_data(f)
        print(f"  Wrote {out_path}  ({len(frames)} frames)")

    print("\n=== summary of all RELEASE events across attempts ===")
    if not all_releases:
        print("  (no releases recorded)")
    else:
        from collections import Counter
        cnt = Counter(r[3] for r in all_releases)
        for reason, n in cnt.most_common():
            print(f"  {reason!r}: {n}")


if __name__ == "__main__":
    main()
