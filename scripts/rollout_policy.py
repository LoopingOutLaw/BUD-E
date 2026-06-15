"""Roll out a trained BUD-E policy in MuJoCo simulation with retry.

Loads a checkpoint, runs PolicyRolloutRunner for N random cube positions,
captures frames + per-try labels, writes an annotated MP4 showing
"try N/M" overlays and SUCCESS/FAILED markers per rollout.

Headless usage:
    unset PYTHONPATH
    MUJOCO_GL=egl PYTHONPATH=src python scripts/rollout_policy.py \
        --ckpt checkpoints/pick_224/pick_224_final.pt \
        --out demos/videos/pick_vla_rollout.mp4 \
        --num-rollouts 5 --img-size 224

Live viewer usage (needs $DISPLAY):
    cd /home/aditya/BUD-E && \
    MUJOCO_GL=glfw DISPLAY=:1 XDG_RUNTIME_DIR=/tmp \
    PYTHONPATH=src python scripts/rollout_policy.py \
        --ckpt /home/aditya/bude_vla/checkpoints/pick_224/pick_224_final.pt \
        --out /home/aditya/BUD-E/demos/videos/pick_vla_rollout_live.mp4 \
        --num-rollouts 5 --img-size 224 --viewer --slow 0.04
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio
import mujoco
import mujoco.viewer  # noqa: F401  ensure module is loaded for launch_passive
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from bude_vla.env_runner import PolicyRolloutRunner
from bude_vla.envs.so101_mjx import ARM_MODEL_PATH
from bude_vla.models.policy import BUDEPolicy, BUDEConfig


def _load_policy(ckpt_path: str, img_size: int, device: str):
    cfg = BUDEConfig()
    cfg.img_size = img_size
    cfg.patch_size = 16
    cfg.chunk_size = 4
    policy = BUDEPolicy(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    loss_hist = ckpt.get("loss_history", [])
    step = ckpt.get("step", "?")
    if loss_hist:
        print(f"  loaded checkpoint step={step}, final loss={loss_hist[-1][1]:.6f}")
    else:
        print(f"  loaded checkpoint step={step}")
    return policy


def _random_cube_positions(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    return [np.array([float(rng.uniform(0.50, 0.75)),
                      float(rng.uniform(-0.15, 0.15))])
            for _ in range(n)]


def _add_overlay(frame: np.ndarray, text: str,
                 status: str = "", img_size: int = 224) -> np.ndarray:
    out = frame.copy()
    font_scale = max(0.4, img_size / 600)
    thickness = max(1, img_size // 300)
    color = (0, 255, 0) if status == "SUCCESS" else (255, 255, 255)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    if status:
        cv2.putText(out, status, (10, int(img_size * 0.92)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1,
                    color, max(2, thickness + 1), cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to .pt checkpoint")
    ap.add_argument("--out", default="demos/videos/pick_vla_rollout.mp4")
    ap.add_argument("--num-rollouts", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-tries", type=int, default=3)
    ap.add_argument("--max-steps-per-try", type=int, default=350)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None,
                    help="cuda or cpu; default auto-detect")
    ap.add_argument("--viewer", action="store_true",
                    help="open a live MuJoCo viewer per rollout (needs DISPLAY). "
                         "Hits env_runner internals via a sync hook — visualizes "
                         "the arm in real time, doesn't replace MP4 frame capture.")
    ap.add_argument("--slow", type=float, default=0.0,
                    help="seconds to sleep per sim step in viewer mode "
                         "(0.04 ~= real-time). Ignored when --viewer is off.")
    args = ap.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"loading policy from {args.ckpt} (device={device})")
    policy = _load_policy(args.ckpt, args.img_size, device)

    model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
    data = mujoco.MjData(model)

    positions = _random_cube_positions(args.num_rollouts, seed=args.seed)
    runner = PolicyRolloutRunner(
        model, img_size=args.img_size,
        max_steps_per_try=args.max_steps_per_try,
        max_tries=args.max_tries,
        device=device,
    )

    if args.viewer and os.environ.get("MUJOCO_GL") != "glfw":
        os.environ["MUJOCO_GL"] = "glfw"

    all_frames: list = []
    n_success = 0

    for i, cube_xy in enumerate(positions):
        print(f"rollout {i + 1}/{args.num_rollouts}: "
              f"cube=({cube_xy[0]:.2f}, {cube_xy[1]:.2f})")
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        data.qpos[0:3] = [cube_xy[0], cube_xy[1], 0.445]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        viewer = None
        if args.viewer:
            try:
                viewer = mujoco.viewer.launch_passive(model, data)
                print(f"  [viewer] live window open on DISPLAY={os.environ.get('DISPLAY', ':0')}")
            except Exception as exc:
                print(f"  [viewer] could not open: {exc}; continuing headless")
                viewer = None

        result = runner.run_one(
            data, policy, cube_xy,
            viewer=viewer, step_delay=args.slow,
        )

        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

        status = "SUCCESS" if result.success else "FAILED"
        n_success += int(result.success)
        print(f"  -> {status} in {result.n_tries} try/tries")

        for frame, label in zip(result.frames, result.try_labels):
            is_success_frame = "SUCCESS" in label
            overlay_status = "SUCCESS" if is_success_frame else ""
            annotated = _add_overlay(
                frame,
                f"#{i + 1} {label}",
                status=overlay_status,
                img_size=args.img_size,
            )
            all_frames.append(annotated)

        if not result.success:
            tail_frame = all_frames[-1].copy()
            cv2.putText(tail_frame, "FAILED", (10, args.img_size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                        cv2.LINE_AA)
            all_frames.append(tail_frame)

    runner.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=30, codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
        macro_block_size=1,
    )
    for f in all_frames:
        writer.append_data(f)
    writer.close()

    rate = n_success / args.num_rollouts * 100 if args.num_rollouts > 0 else 0
    print(f"\n=== DONE  {n_success}/{args.num_rollouts} success " f"({rate:.0f}%) ===")
    print(f"  MP4: {out_path}  ({len(all_frames)} frames)")


if __name__ == "__main__":
    main()
