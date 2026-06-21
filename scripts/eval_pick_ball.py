"""Eval a trained pick policy — physics-only grasping (matches training exactly).

Matches training conditions:
 - top-down wrist pose (wrist_flex=pi/2, wrist_roll=pi/2)
 - physics-only grasping (NO GraspController teleport)
 - cube at z=CUBE_REST_Z (0.015)
 - wrist_cam (not pov)
 - cube spawn range (0.15-0.35, -0.10-0.10) matching training
 - 9D proprio [arm(5)+gripper(1)+target_rel(2)+is_grasping(1)] for pick_v10 checkpoints
 - 7D proprio [arm(5)+gripper(1)+is_grasping(1)] for pick_v8/v9 checkpoints
 - 6D proprio [arm(5)+gripper(1)] for older pick_v7 checkpoints
 - kinematic arm execution + physics gripper (matches training action format)
 - MAX_STEPS = 4000 for full pick-and-place sequence

Usage:
    unset PYTHONPATH
    MUJOCO_GL=egl PYTHONPATH=src python scripts/eval_pick_ball.py \
        --ckpt checkpoints/pick_v10/pick_v10_step_050000.pt \
        --num-episodes 10
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio
import mujoco
import numpy as np
import torch
from pathlib import Path

from bude_vla.data.action_normalization import denormalize_actions
from bude_vla.data.lerobot_v3 import _tokenize_instruction
from bude_vla.envs.so101_mjx import (
    ARM_QPOS_START, ARM_QPOS_END,
    GRIPPER_QPOS_START, GRIPPER_QPOS_END,
    CUBE_QPOS_START, CUBE_QPOS_END,
    N_ARM_JOINTS,
    load_arm_model, CUBE_REST_Z,
    is_grasping_from_contacts,
)
from bude_vla.models.policy import BUDEPolicy, BUDEConfig

SUCCESS_THRESHOLD = 0.05
INSTRUCTION = "pick up the red cube and place it in the blue target zone"
MAX_STEPS = 4000
SUBSTEPS_PER_FRAME = 4  # must match training (record_pick_episodes.py)


def load_policy(ckpt_path: str, img_size: int, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = BUDEConfig()

    saved_cfg = ckpt.get("config", {})
    cfg.use_dinov2 = saved_cfg.get("use_dinov2", False)
    cfg.use_minilm = saved_cfg.get("use_minilm", False)
    cfg.n_history_frames = saved_cfg.get("n_history_frames", 1)
    cfg.chunk_size = saved_cfg.get("chunk_size", 4)
    cfg.img_size = saved_cfg.get("img_size", img_size)

    action_lo = ckpt.get("action_norm_lo", None)
    action_hi = ckpt.get("action_norm_hi", None)
    if action_lo is not None:
        action_lo = np.asarray(action_lo, dtype=np.float32)
        action_hi = np.asarray(action_hi, dtype=np.float32)
        cfg.action_dim = len(action_lo)
        cfg.state_dim = saved_cfg.get("state_dim", len(action_lo))

    cfg.patch_size = 16
    policy = BUDEPolicy(cfg).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    step = ckpt.get("step", "?")
    loss_hist = ckpt.get("loss_history", [])
    final_loss = loss_hist[-1][1] if loss_hist else float("nan")
    print(f" loaded step={step}, final_loss={final_loss:.6f}, "
          f"action_dim={cfg.action_dim}, state_dim={cfg.state_dim}")
    return policy, action_lo, action_hi, cfg


def reset_cube(data, cx: float, cy: float):
    data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [cx, cy, CUBE_REST_Z]
    data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[CUBE_QPOS_START:CUBE_QPOS_END] = 0.0
    mujoco.mj_forward(data.model, data)


def reset_arm(model, data):
    """Top-down initial pose (EXACTLY matches training)."""
    data.qpos[ARM_QPOS_START:ARM_QPOS_END] = np.array([
        0.0,           # shoulder_pan
        -0.5,          # shoulder_lift
        0.95,          # elbow_flex
        np.pi / 2,     # wrist_flex = 90° (top-down)
        np.pi / 2,     # wrist_roll = 90° (fingers along Y)
    ])
    data.qpos[GRIPPER_QPOS_START:GRIPPER_QPOS_END] = 0.3  # partially open
    data.qvel[ARM_QPOS_START:GRIPPER_QPOS_END] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def is_success(data) -> bool:
    cube_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    err = np.linalg.norm(data.xpos[cube_id, :2] - data.xpos[target_id, :2])
    return err < SUCCESS_THRESHOLD


def is_failure(data, step) -> bool:
    if step >= MAX_STEPS:
        return True
    cube_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_pos = data.xpos[cube_id]
    if np.any(np.isnan(cube_pos)):
        return True
    if cube_pos[2] < -0.05 or cube_pos[2] > 1.5:
        return True
    if np.any(np.abs(data.qpos[ARM_QPOS_START:ARM_QPOS_END]) > 3.5):
        return True
    return False


def build_batch(image: np.ndarray, proprio: np.ndarray,
                text_ids: np.ndarray, device: str,
                n_history_frames: int = 1) -> dict:
    # For n_history_frames > 1: image is already stacked (H, W, n_h*6)
    # For n_history_frames == 1: image is just (H, W, 6)
    img = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0
    return {
        "images": img.unsqueeze(0).to(device),
        "text_ids": torch.from_numpy(text_ids).unsqueeze(0).to(device),
        "instruction": [INSTRUCTION],
        "proprio": torch.from_numpy(proprio.astype(np.float32)).unsqueeze(0).to(device),
        "domain_id": torch.tensor([0], dtype=torch.long).to(device),
    }


def add_overlay(frame: np.ndarray, text: str, status: str = "",
                grasped: bool = False) -> np.ndarray:
    out = frame.copy()
    if out.shape[-1] >= 3 and out.shape[-1] != 3:
        out = np.ascontiguousarray(out[..., :3])
    color = (0, 255, 0) if status == "SUCCESS" else \
            (0, 0, 255) if status == "FAILED" else (255, 255, 255)
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if grasped:
        cv2.putText(out, "GRASP", (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 220, 255), 1, cv2.LINE_AA)
    if status:
        cv2.putText(out, status, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return out


def run_eval(policy, model, data, obs_renderer, vid_renderer, text_ids,
             action_lo, action_hi, cfg, device,
             num_episodes, seed):
    rng = np.random.default_rng(seed)
    all_frames = []
    n_success = 0

    front_top_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_top")
    wrist_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    portfolio_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "portfolio")
    vid_cam = portfolio_cam if portfolio_cam >= 0 else wrist_cam

    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    gripperframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    use_9d = (cfg.state_dim == 9)
    use_contact_signal = (cfg.state_dim >= 7)
    n_h = cfg.n_history_frames
    C_single = 6  # single-frame dual-cam channels

    for ep in range(num_episodes):
        cx = float(rng.uniform(0.15, 0.35))
        cy = float(rng.uniform(-0.10, 0.10))
        print(f" ep {ep:3d} cube=({cx:.2f},{cy:.2f})", end=" ", flush=True)

        mujoco.mj_resetData(model, data)
        reset_arm(model, data)
        reset_cube(data, cx, cy)

        # Let cube settle (matches training)
        for _ in range(50):
            mujoco.mj_step(model, data)

        chunk = None
        cursor = 0
        ever_grasped = False
        img_buffer = []  # reset per episode

        for step in range(MAX_STEPS):
            # Render from SAME cameras as training
            obs_renderer.update_scene(data, camera=front_top_cam)
            img_top = np.asarray(obs_renderer.render()).copy()
            obs_renderer.update_scene(data, camera=wrist_cam)
            img_wrist = np.asarray(obs_renderer.render()).copy()
            img = np.concatenate([img_top, img_wrist], axis=-1)

            # Frame stacking for n_history_frames (matches training exactly)
            img_buffer.append(img)
            if len(img_buffer) > n_h:
                img_buffer = img_buffer[-n_h:]
            if n_h <= 1:
                stacked_img = img
            else:
                # Pad early frames by repeating the first captured frame
                while len(img_buffer) < n_h:
                    img_buffer.insert(0, img_buffer[0])
                window = np.stack(img_buffer, axis=0)  # (n_h, H, W, C_single)
                stacked_img = window.reshape(window.shape[1], window.shape[2],
                                            n_h * C_single)  # (H, W, n_h*6)

            vid_renderer.update_scene(data, camera=vid_cam)
            vid_frame = np.asarray(vid_renderer.render()).copy()

            # Proprio: 9D [arm(5)+gripper(1)+target_rel(2)+is_grasping(1)] or 7D or 6D
            # is_grasping uses SAME contact helper as recording — no heuristic mismatch
            gripper_pos = data.site_xpos[gripperframe_id]
            is_grasping = is_grasping_from_contacts(model, data)
            if is_grasping > 0.5:
                ever_grasped = True

            if use_9d:
                target_pos = data.xpos[target_body_id]
                target_rel = target_pos[:2] - gripper_pos[:2]
                arm_proprio = np.concatenate([
                    data.qpos[ARM_QPOS_START:GRIPPER_QPOS_START + 1],  # 6D
                    target_rel,                                         # 2D
                    [is_grasping],                                      # 1D = 9D
                ]).astype(np.float32)
            elif use_contact_signal:
                arm_proprio = np.concatenate([
                    data.qpos[ARM_QPOS_START:GRIPPER_QPOS_START + 1],  # 6D
                    [is_grasping],                                      # 1D = 7D
                ]).astype(np.float32)
            else:
                arm_proprio = data.qpos[ARM_QPOS_START:GRIPPER_QPOS_END].astype(np.float32).copy()

            batch = build_batch(stacked_img, arm_proprio, text_ids, device,
                                n_history_frames=n_h)
            if chunk is None or cursor >= cfg.chunk_size:
                chunk = policy.sample(batch)[0].detach().cpu().numpy()
                cursor = 0
            a = chunk[cursor]
            cursor += 1

            if action_lo is not None:
                a = denormalize_actions(a, action_lo, action_hi)

            if not np.any(np.isnan(a)):
                arm_target = np.clip(a[:N_ARM_JOINTS], -3.5, 3.5).astype(np.float64)
                gripper_ctrl = float(np.clip(a[N_ARM_JOINTS], -1.5, 1.5))
            else:
                arm_target = np.array([0.0, -0.5, 0.95, np.pi/2, np.pi/2])
                gripper_ctrl = 0.3

            # Execute: kinematic arm + physics gripper (matches training EXACTLY)
            # Training sets data.ctrl[:] from policy, where ctrl[0:5] are always 0.0
            # and ctrl[5] is the gripper actuator value.
            data.qpos[ARM_QPOS_START:ARM_QPOS_END] = arm_target
            data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0.0
            data.ctrl[:N_ARM_JOINTS] = 0.0          # arm actuators OFF (kinematic)
            data.ctrl[N_ARM_JOINTS] = gripper_ctrl  # only gripper actuator

            for _ in range(SUBSTEPS_PER_FRAME):
                mujoco.mj_step(model, data)

            frame = add_overlay(vid_frame, f"ep {ep} step {step}",
                                grasped=(use_contact_signal and is_grasping > 0.5))
            all_frames.append(frame)

            if is_success(data) and ever_grasped:
                n_success += 1
                for _ in range(30):
                    mujoco.mj_step(model, data)
                    vid_renderer.update_scene(data, camera=vid_cam)
                    vf = np.asarray(vid_renderer.render()).copy()
                    f = add_overlay(vf, f"ep {ep}", "SUCCESS", grasped=True)
                    all_frames.append(f)
                print(f"SUCCESS (step {step})")
                break

            if is_failure(data, step):
                f = add_overlay(vid_frame, f"ep {ep}", "FAILED")
                all_frames.append(f)
                print(f"FAILED (step {step})")
                break

        else:
            f = add_overlay(vid_frame, f"ep {ep}", "FAILED")
            all_frames.append(f)
            cube_final = data.xpos[cube_body_id].copy()
            target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
            target_pos = data.xpos[target_id, :2].copy()
            dist = np.linalg.norm(cube_final[:2] - target_pos)
            print(f"FAILED (timeout) cube=[{cube_final[0]:.3f},{cube_final[1]:.3f}] "
                  f"grasped={ever_grasped} dist={dist:.3f}")

    rate = n_success / num_episodes * 100 if num_episodes > 0 else 0
    print(f"\n=== EVAL {n_success}/{num_episodes} success ({rate:.0f}%) ===")
    return n_success, num_episodes, all_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--video-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="demos/videos/eval_pick_v8.mp4")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    policy, action_lo, action_hi, cfg = load_policy(args.ckpt, args.img_size, device)
    model = load_arm_model()
    data = mujoco.MjData(model)
    obs_renderer = mujoco.Renderer(model, height=args.img_size, width=args.img_size)
    vid_renderer = mujoco.Renderer(model, height=args.video_size, width=args.video_size)
    text_ids = _tokenize_instruction(INSTRUCTION)

    print(f"Running {args.num_episodes} eval episodes "
          f"(obs={args.img_size}x{args.img_size}, video={args.video_size}x{args.video_size})...")
    n_success, n_total, frames = run_eval(
        policy, model, data, obs_renderer, vid_renderer, text_ids,
        action_lo, action_hi, cfg, device,
        args.num_episodes, args.seed,
    )

    rate = n_success / n_total * 100 if n_total > 0 else 0
    print(f"\n=== EVAL {n_success}/{n_total} success ({rate:.0f}%) ===")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=30, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p"],
                                macro_block_size=1)
    for f in frames:
        if f.shape[-1] == 6:
            f = np.ascontiguousarray(f[..., :3])
        writer.append_data(f)
    writer.close()
    print(f" MP4: {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
