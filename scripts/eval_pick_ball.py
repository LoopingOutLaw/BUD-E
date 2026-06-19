"""Eval a trained pick_v4_ball policy in MuJoCo — rollout + success-rate + MP4.

Usage:
    unset PYTHONPATH
    MUJOCO_GL=egl python scripts/eval_pick_ball.py \
        --ckpt checkpoints/pick_v4_ball/pick_v4_ball_final.pt \
        --num-episodes 20

GRASP FIX (see src/bude_vla/grasp.py for full rationale):
Grasp attach/carry/release used to be done with `attach_offset`/`carry_ball`
helpers defined in this file, which captured the gripper-ball offset the
instant gripper_ctrl crossed -0.1 -- regardless of whether the gripper was
actually anywhere near the ball -- and replayed that exact (possibly large)
offset every frame after. That's what produced a visible floating gap
between the gripper and the ball in rollout videos. Both helpers are
replaced by `bude_vla.grasp.GraspController`, which only attaches once the
ball is geometrically enclosed by the jaw AND in real MuJoCo contact AND
that holds for several consecutive steps, snapping flush (zero gap) at the
moment of attach. The video overlay now also prints "GRASP" whenever the
controller considers the ball genuinely held, so you can visually confirm
the fix from the rendered output, not just trust the success metric.
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
from bude_vla.grasp import GraspController
from bude_vla.envs.so101_mjx import (
    ARM_QPOS_START, ARM_QPOS_END,
    GRIPPER_QPOS_START, GRIPPER_QPOS_END,
    CUBE_QPOS_START, CUBE_QPOS_END,
    N_ARM_JOINTS,
    load_arm_model, default_joint_angles,
)
from bude_vla.models.policy import BUDEPolicy, BUDEConfig

BALL_RADIUS = 0.0125
GROUND_Z = 0.0295
SUCCESS_THRESHOLD = 0.05
INSTRUCTION = "pick up the red ball and place it in the blue target zone"
DOMAIN_ID = 0
MAX_STEPS = 500


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
        cfg.state_dim = len(action_lo)

    cfg.patch_size = 16
    policy = BUDEPolicy(cfg).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    step = ckpt.get("step", "?")
    loss_hist = ckpt.get("loss_history", [])
    final_loss = loss_hist[-1][1] if loss_hist else float("nan")
    print(f"  loaded step={step}, final_loss={final_loss:.6f}, "
          f"action_dim={cfg.action_dim}, state_dim={cfg.state_dim}")
    return policy, action_lo, action_hi, cfg


def reset_ball(data, cx: float, cy: float):
    data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [cx, cy, GROUND_Z]
    data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[CUBE_QPOS_START:CUBE_QPOS_END] = 0.0
    mujoco.mj_forward(data.model, data)


def reset_arm(model, data):
    data.qpos[ARM_QPOS_START:ARM_QPOS_END] = default_joint_angles(model)
    data.qpos[GRIPPER_QPOS_START:GRIPPER_QPOS_END] = 1.5
    data.qvel[ARM_QPOS_START:GRIPPER_QPOS_END] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def ball_xyz(data) -> np.ndarray:
    cube_body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    return data.xpos[cube_body_id].copy()


def target_xy(data) -> np.ndarray:
    target_body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    return data.xpos[target_body_id, :2].copy()


def is_success(data) -> bool:
    cube_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    err = np.linalg.norm(data.xpos[cube_id, :2] - data.xpos[target_id, :2])
    return err < SUCCESS_THRESHOLD


def is_failure(data, step) -> bool:
    if step >= MAX_STEPS:
        return True
    cube_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    ball = data.xpos[cube_id]
    if np.any(np.isnan(ball)):
        return True
    if ball[2] < -0.05 or ball[2] > 1.5:
        return True
    if np.any(np.abs(data.qpos[ARM_QPOS_START:ARM_QPOS_END]) > 3.5):
        return True
    return False


def build_batch(image: np.ndarray, proprio: np.ndarray,
                text_ids: np.ndarray, device: str) -> dict:
    img = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0
    return {
        "images": img.unsqueeze(0).to(device),
        "text_ids": torch.from_numpy(text_ids).unsqueeze(0).to(device),
        "instruction": [INSTRUCTION],
        "proprio": torch.from_numpy(proprio.astype(np.float32)).unsqueeze(0).to(device),
        "domain_id": torch.tensor([DOMAIN_ID], dtype=torch.long).to(device),
    }


def add_overlay(frame: np.ndarray, text: str, status: str = "",
                grasped: bool = False) -> np.ndarray:
    out = frame.copy()
    if out.shape[-1] >= 3 and out.shape[-1] != 3:
        out = np.ascontiguousarray(out[..., :3])
    color = (0, 255, 0) if status == "SUCCESS" else (0, 0, 255) if status == "FAILED" else (255, 255, 255)
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if grasped:
        cv2.putText(out, "GRASP", (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 220, 255), 1, cv2.LINE_AA)
    if status:
        cv2.putText(out, status, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return out


def run_eval(policy, model, data, obs_renderer, vid_renderer, text_ids,
             action_lo, action_hi, device, img_size,
             num_episodes, seed):
    rng = np.random.default_rng(seed)
    all_frames = []
    successes = []
    n_success = 0
    n_grasped_at_all = 0

    front_top_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_top")
    pov_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "pov")

    grasp_ctrl = GraspController(model)

    for ep in range(num_episodes):
        cx = float(rng.uniform(0.28, 0.32))
        cy = float(rng.uniform(-0.02, 0.02))
        print(f"  ep {ep:3d}  ball=({cx:.2f},{cy:.2f})", end="  ", flush=True)

        mujoco.mj_resetData(model, data)
        reset_arm(model, data)
        reset_ball(data, cx, cy)
        grasp_ctrl.reset()
        ever_grasped_this_ep = False

        chunk = None
        cursor = 0

        for step in range(MAX_STEPS):
            obs_renderer.update_scene(data, camera=front_top_cam)
            img_top = np.asarray(obs_renderer.render()).copy()
            obs_renderer.update_scene(data, camera=pov_cam)
            img_pov = np.asarray(obs_renderer.render()).copy()
            img = np.concatenate([img_top, img_pov], axis=-1)

            vid_renderer.update_scene(data, camera=pov_cam)
            vid_frame = np.asarray(vid_renderer.render()).copy()

            arm_proprio = data.qpos[ARM_QPOS_START:GRIPPER_QPOS_END].astype(np.float32).copy()

            batch = build_batch(img, arm_proprio, text_ids, device)
            if chunk is None or cursor >= chunk.shape[0]:
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
                arm_target = default_joint_angles(model)
                gripper_ctrl = 1.5

            data.qpos[ARM_QPOS_START:ARM_QPOS_END] = arm_target
            data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0.0
            data.ctrl[N_ARM_JOINTS] = gripper_ctrl

            # Physically-gated grasp bookkeeping: attaches only on real
            # enclosure + contact + debounce, carries flush, releases on
            # real jaw reopening or excess drift. See grasp.py.
            grasp_ctrl.update(model, data, jaw_qpos=float(data.qpos[GRIPPER_QPOS_START]))
            if grasp_ctrl.state.attached:
                ever_grasped_this_ep = True

            mujoco.mj_step(model, data)

            frame = add_overlay(vid_frame, f"ep {ep} step {step}",
                               grasped=grasp_ctrl.state.attached)
            all_frames.append(frame)

            if is_success(data):
                n_success += 1
                successes.append(True)
                for _ in range(30):
                    grasp_ctrl.update(model, data, jaw_qpos=float(data.qpos[GRIPPER_QPOS_START]))
                    mujoco.mj_step(model, data)
                    vid_renderer.update_scene(data, camera=pov_cam)
                    vf = np.asarray(vid_renderer.render()).copy()
                    f = add_overlay(vf, f"ep {ep}", "SUCCESS",
                                    grasped=grasp_ctrl.state.attached)
                    all_frames.append(f)
                print(f"SUCCESS (step {step})", end="  ")
                break

            if is_failure(data, step):
                successes.append(False)
                f = add_overlay(vid_frame, f"ep {ep}", "FAILED")
                all_frames.append(f)
                print(f"FAILED (step {step})", end="  ")
                break
        else:
            successes.append(False)
            print("FAILED (timeout)", end="  ")
            f = add_overlay(vid_frame, f"ep {ep}", "FAILED")
            all_frames.append(f)

        if ever_grasped_this_ep:
            n_grasped_at_all += 1
        print(f"[grasped_at_all={ever_grasped_this_ep}]")

    print(f"\n  diagnostic: ball was physically grasped (real enclosure+contact) "
          f"in {n_grasped_at_all}/{num_episodes} episodes")
    return n_success, num_episodes, all_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--video-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="demos/videos/eval_pick_v4_ball.mp4")
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
        action_lo, action_hi, device, args.img_size,
        args.num_episodes, args.seed,
    )

    rate = n_success / n_total * 100 if n_total > 0 else 0
    print(f"\n=== EVAL  {n_success}/{n_total} success ({rate:.0f}%) ===")

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
    print(f"  MP4: {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
