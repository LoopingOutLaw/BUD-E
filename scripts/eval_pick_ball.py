"""Eval a trained pick policy — physics-only grasping (matches training exactly).

Matches training conditions:
 - top-down wrist pose (wrist_flex=pi/2, wrist_roll=pi/2)
 - physics-only grasping (NO GraspController teleport)
 - cube at z=CUBE_REST_Z (0.015)
 - wrist_cam (not pov)
 - cube spawn range (0.15-0.35, -0.10-0.10) matching training
 - 10D proprio [arm(5)+gripper(1)+target_rel(2)+any_contact(1)+is_grasping(1)] for contact-aware checkpoints
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

from bude_vla.action_space import apply_policy_action, make_ik_controller
from bude_vla.data.action_normalization import denormalize_actions
from bude_vla.data.lerobot_v3 import _tokenize_instruction, _domain_from_instruction
from bude_vla.envs.so101_mjx import (
    ARM_QPOS_START, ARM_QPOS_END,
    GRIPPER_QPOS_START, GRIPPER_QPOS_END,
    CUBE_QPOS_START, CUBE_QPOS_END,
    N_ARM_JOINTS,
    load_arm_model, CUBE_REST_Z,
    is_grasping_from_contacts,
    is_touching_cube_from_contacts,
    build_pick_proprio,
)
from bude_vla.models.policy import BUDEPolicy, BUDEConfig
from bude_vla.perception import detect_red_centroid

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
    cfg.use_bc_head = saved_cfg.get("use_bc_head", False)
    cfg.use_visual_action_cond = saved_cfg.get("use_visual_action_cond", False)
    cfg.use_context_action_head = saved_cfg.get("use_context_action_head", False)
    cfg.use_perception = saved_cfg.get("use_perception", False)
    cfg.use_perception_action_cond = saved_cfg.get("use_perception_action_cond", False)
    cfg.perception_dim = saved_cfg.get("perception_dim", 3)
    cfg.use_gripper_trigger_head = saved_cfg.get("use_gripper_trigger_head", False)
    cfg.gripper_trigger_threshold = saved_cfg.get("gripper_trigger_threshold", 0.5) or 0.5
    cfg.gripper_trigger_close_value = saved_cfg.get("gripper_trigger_close_value", -1.0) or -1.0
    cfg.action_space = saved_cfg.get("action_space", "joint_abs")
    cfg.ee_delta_scale = saved_cfg.get("ee_delta_scale", 0.05) or 0.05

    action_lo = ckpt.get("action_norm_lo", None)
    action_hi = ckpt.get("action_norm_hi", None)
    if action_lo is not None:
        action_lo = np.asarray(action_lo, dtype=np.float32)
        action_hi = np.asarray(action_hi, dtype=np.float32)
        cfg.action_dim = len(action_lo)
        cfg.state_dim = saved_cfg.get("state_dim", len(action_lo))

    cfg.patch_size = 16
    policy = BUDEPolicy(cfg).to(device)
    ema_sd = ckpt.get("ema_state_dict")
    if ema_sd is not None:
        policy.load_state_dict(ema_sd)
        weight_src = "EMA"
    else:
        policy.load_state_dict(ckpt["model_state_dict"])
        weight_src = "raw"
    policy.eval()

    step = ckpt.get("step", "?")
    loss_hist = ckpt.get("loss_history", [])
    final_loss = loss_hist[-1][1] if loss_hist else float("nan")
    print(f" loaded step={step} ({weight_src} weights), final_loss={final_loss:.6f}, "
          f"action_dim={cfg.action_dim}, state_dim={cfg.state_dim}, action_space={cfg.action_space}")
    return policy, action_lo, action_hi, cfg


def parse_cube_positions(spec: str | None) -> list[tuple[float, float]] | None:
    """Parse explicit eval cube positions formatted as "x,y;x,y"."""
    if spec is None or not spec.strip():
        return None
    positions: list[tuple[float, float]] = []
    for raw_pair in spec.split(";"):
        pair = raw_pair.strip()
        if not pair:
            continue
        parts = [part.strip() for part in pair.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid cube position {pair!r}; expected x,y pairs separated by semicolons"
            )
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid cube position {pair!r}; x,y values must be numbers"
            ) from exc
        positions.append((x, y))
    if not positions:
        raise ValueError("--cube-positions did not contain any x,y pairs")
    return positions


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
    data.ctrl[ARM_QPOS_START:ARM_QPOS_END] = data.qpos[ARM_QPOS_START:ARM_QPOS_END]  # hold pose during settle
    data.ctrl[GRIPPER_QPOS_START] = 0.3
    mujoco.mj_forward(model, data)


def is_success(data) -> bool:
    cube_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    err = np.linalg.norm(data.xpos[cube_id, :2] - data.xpos[target_id, :2])
    return err < SUCCESS_THRESHOLD


def is_failure(data, step, max_steps: int = MAX_STEPS) -> bool:
    if step >= max_steps:
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
    perception = detect_red_centroid(image, n_history_frames=n_history_frames)
    img = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0
    return {
        "images": img.unsqueeze(0).to(device),
        "text_ids": torch.from_numpy(text_ids).unsqueeze(0).to(device),
        "instruction": [INSTRUCTION],
        "proprio": torch.from_numpy(proprio.astype(np.float32)).unsqueeze(0).to(device),
        "perception": torch.from_numpy(perception).unsqueeze(0).to(device),
        "domain_id": torch.tensor([_domain_from_instruction(INSTRUCTION)], dtype=torch.long).to(device),
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
             num_episodes, seed,
             ensembling: bool = False, ensembling_k: float = 0.5,
             replan_every: int = 1,
             exec_first_only: bool = False,
             debug_actions: bool = False,
             contact_close_reflex: bool = False,
             contact_close_steps: int = 120,
             contact_close_value: float = -1.0,
             cube_positions: list[tuple[float, float]] | None = None,
             cube_x_range: tuple[float, float] = (0.15, 0.35),
             cube_y_range: tuple[float, float] = (-0.10, 0.10),
             max_steps: int = MAX_STEPS):
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

    use_9d_or_10d = (cfg.state_dim in (9, 10))
    use_contact_signal = (cfg.state_dim >= 7)
    n_h = cfg.n_history_frames
    C_single = 6  # single-frame dual-cam channels

    for ep in range(num_episodes):
        if cube_positions:
            cx, cy = cube_positions[ep % len(cube_positions)]
        else:
            cx = float(rng.uniform(cube_x_range[0], cube_x_range[1]))
            cy = float(rng.uniform(cube_y_range[0], cube_y_range[1]))
        print(f" ep {ep:3d} cube=({cx:.2f},{cy:.2f})", end=" ", flush=True)

        mujoco.mj_resetData(model, data)
        reset_arm(model, data)
        reset_cube(data, cx, cy)

        # Let cube settle (matches training)
        for _ in range(50):
            mujoco.mj_step(model, data)

        chunk = None
        cursor = 0
        action_queue: list = []  # only used when ensembling=True
        ever_grasped = False
        close_until = -1
        ik = make_ik_controller(model, data) if cfg.action_space == "ee_delta" else None
        img_buffer = []  # reset per episode

        for step in range(max_steps):
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
                window = np.ascontiguousarray(window)
                stacked_img = np.transpose(window, (1, 2, 0, 3)).reshape(
                    window.shape[1], window.shape[2], n_h * C_single)

            vid_renderer.update_scene(data, camera=vid_cam)
            vid_frame = np.asarray(vid_renderer.render()).copy()

            # Proprio builder is shared with recording/training to avoid layout drift.
            is_touching = is_touching_cube_from_contacts(model, data)
            is_grasping = is_grasping_from_contacts(model, data)
            if is_touching > 0.5 and contact_close_reflex:
                close_until = max(close_until, step + contact_close_steps)
            if is_grasping > 0.5:
                ever_grasped = True
            arm_proprio = build_pick_proprio(model, data, cfg.state_dim)

            batch = build_batch(stacked_img, arm_proprio, text_ids, device,
                                n_history_frames=n_h)
            if ensembling:
                # Replan every `replan_every` steps (default: every step) and
                # blend with whatever's left in the queue instead of blindly
                # committing to a full chunk_size-step open-loop sequence.
                if not action_queue or step % max(1, replan_every) == 0:
                    new_chunk = policy.sample(batch)[0].detach().cpu().numpy()
                    if action_lo is not None:
                        new_chunk = denormalize_actions(new_chunk, action_lo, action_hi)
                    q = list(action_queue)
                    for i, new_a in enumerate(new_chunk):
                        if i < len(q):
                            q[i] = ensembling_k * q[i] + (1 - ensembling_k) * new_a
                        else:
                            q.append(new_a)
                    action_queue = q
                a = action_queue.pop(0)
            else:
                if exec_first_only:
                    chunk = policy.sample(batch)[0].detach().cpu().numpy()
                    a = chunk[0]
                    if action_lo is not None:
                        a = denormalize_actions(a, action_lo, action_hi)
                else:
                    if chunk is None or cursor >= cfg.chunk_size:
                        chunk = policy.sample(batch)[0].detach().cpu().numpy()
                        cursor = 0
                    a = chunk[cursor]
                    cursor += 1
                    if action_lo is not None:
                        a = denormalize_actions(a, action_lo, action_hi)

            if not np.any(np.isnan(a)):
                arm_target, gripper_ctrl = apply_policy_action(
                    model, data, a, cfg, ik=ik,
                    contact_close_reflex=contact_close_reflex,
                    close_active=step <= close_until,
                    contact_close_value=contact_close_value,
                )
                if debug_actions and (step < 20 or step % 50 == 0):
                    print(
                        f"    step {step:04d} action={np.array2string(np.asarray(a), precision=3)} "
                        f"arm={np.array2string(arm_target, precision=3)} grip={gripper_ctrl:+.3f}",
                        flush=True,
                    )
            else:
                data.ctrl[:N_ARM_JOINTS] = np.array([0.0, -0.5, 0.95, np.pi/2, np.pi/2])
                data.ctrl[N_ARM_JOINTS] = 0.3

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

            if is_failure(data, step, max_steps=max_steps):
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
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--video-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="demos/videos/eval_pick_v8.mp4")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--ensembling", action="store_true",
                     help="Replan every step (or every --replan-every steps) and "
                          "blend overlapping chunk predictions instead of blindly "
                          "executing a full chunk_size-step open-loop sequence. "
                          "Usually improves precision-critical tasks like grasping.")
    ap.add_argument("--ensembling-k", type=float, default=0.5,
                     help="Weight given to the OLD (already-queued) prediction when "
                          "blending with a fresh chunk. 0=only trust the new chunk, "
                          "1=ignore new chunk entirely.")
    ap.add_argument("--replan-every", type=int, default=1,
                    help="With --ensembling, how often (in steps) to query the "
                         "policy for a new chunk. 1 = every step (most reactive, "
                         "most compute).")
    ap.add_argument("--exec-first-only", action="store_true",
                    help="Drop all but chunk[0] of each sampled chunk and re-sample "
                         "every step. Equivalent to chunk_size=1 without retraining.")
    ap.add_argument("--debug-actions", action="store_true",
                    help="Print pre-clip arm targets during eval to diagnose joint-limit clipping.")
    ap.add_argument("--contact-close-reflex", action="store_true",
                    help="Robot-side reflex: close/hold gripper briefly after any-pad cube contact.")
    ap.add_argument("--contact-close-steps", type=int, default=120)
    ap.add_argument("--contact-close-value", type=float, default=-1.0)
    ap.add_argument("--cube-positions", default=None,
                    help="Explicit eval cube positions as 'x,y;x,y'. Repeats if "
                         "--num-episodes is larger than the list.")
    ap.add_argument("--cube-x-range", nargs=2, type=float, default=(0.15, 0.35),
                    metavar=("MIN", "MAX"),
                    help="Random eval cube x range used when --cube-positions is unset.")
    ap.add_argument("--cube-y-range", nargs=2, type=float, default=(-0.10, 0.10),
                    metavar=("MIN", "MAX"),
                    help="Random eval cube y range used when --cube-positions is unset.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    policy, action_lo, action_hi, cfg = load_policy(args.ckpt, args.img_size, device)
    if cfg.img_size != args.img_size:
        print(f" WARNING: --img-size {args.img_size} does not match checkpoint's "
              f"training resolution cfg.img_size={cfg.img_size}. Overriding to "
              f"{cfg.img_size} to match training exactly (mismatched resolution "
              f"is a common cause of silent 0% eval success).")
    obs_img_size = cfg.img_size  # ALWAYS render observations at training resolution
    model = load_arm_model()
    data = mujoco.MjData(model)
    obs_renderer = mujoco.Renderer(model, height=obs_img_size, width=obs_img_size)
    vid_renderer = mujoco.Renderer(model, height=args.video_size, width=args.video_size)
    text_ids = _tokenize_instruction(INSTRUCTION)

    cube_positions = parse_cube_positions(args.cube_positions)
    if cube_positions:
        print(f"Using fixed eval cube positions: {cube_positions}")
    print(f"Running {args.num_episodes} eval episodes "
          f"(obs={args.img_size}x{args.img_size}, video={args.video_size}x{args.video_size})...")
    n_success, n_total, frames = run_eval(
        policy, model, data, obs_renderer, vid_renderer, text_ids,
        action_lo, action_hi, cfg, device,
        args.num_episodes, args.seed,
        ensembling=args.ensembling, ensembling_k=args.ensembling_k,
        replan_every=args.replan_every,
        exec_first_only=args.exec_first_only,
        debug_actions=args.debug_actions,
        contact_close_reflex=args.contact_close_reflex,
        contact_close_steps=args.contact_close_steps,
        contact_close_value=args.contact_close_value,
        cube_positions=cube_positions,
        cube_x_range=tuple(args.cube_x_range),
        cube_y_range=tuple(args.cube_y_range),
        max_steps=args.max_steps,
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
