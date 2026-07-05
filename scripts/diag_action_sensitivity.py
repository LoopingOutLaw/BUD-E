"""Check whether a pick checkpoint conditions actions on cube image position.

This is the fast regression for the collapse we observed: the arm repeatedly
moved to one fixed pose regardless of cube position. The policy is fed only
camera pixels, language, and proprio. The printed perception token is the
pixel-derived red-cube centroid, not MuJoCo cube state.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from bude_vla.data.action_normalization import denormalize_actions
from bude_vla.data.lerobot_v3 import _domain_from_instruction, _tokenize_instruction
from bude_vla.envs.so101_mjx import (
    ARM_QPOS_START,
    ARM_QPOS_END,
    GRIPPER_QPOS_START,
    GRIPPER_QPOS_END,
    CUBE_QPOS_START,
    CUBE_QPOS_END,
    CUBE_REST_Z,
    load_arm_model,
    is_grasping_from_contacts,
)
from bude_vla.models.policy import BUDEConfig, BUDEPolicy
from bude_vla.perception import detect_red_centroid

INSTRUCTION = "pick up the red cube and place it in the blue target zone"


def load_policy(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved = ckpt.get("config", {})
    cfg = BUDEConfig()
    cfg.use_dinov2 = saved.get("use_dinov2", False)
    cfg.use_minilm = saved.get("use_minilm", False)
    cfg.dinov2_finetune_blocks = saved.get("dinov2_finetune_blocks", 4)
    cfg.n_history_frames = saved.get("n_history_frames", 1)
    cfg.chunk_size = saved.get("chunk_size", 4)
    cfg.img_size = saved.get("img_size", 224)
    cfg.action_dim = saved.get("action_dim", 6)
    cfg.state_dim = saved.get("state_dim", 6)
    cfg.use_bc_head = saved.get("use_bc_head", False)
    cfg.use_visual_action_cond = saved.get("use_visual_action_cond", False)
    cfg.use_context_action_head = saved.get("use_context_action_head", False)
    cfg.use_perception = saved.get("use_perception", False)
    cfg.perception_dim = saved.get("perception_dim", 3)
    cfg.patch_size = 16

    policy = BUDEPolicy(cfg).to(device)
    state_dict = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    policy.load_state_dict(state_dict)
    policy.eval()

    lo = ckpt.get("action_norm_lo")
    hi = ckpt.get("action_norm_hi")
    lo = np.asarray(lo, dtype=np.float32) if lo is not None else None
    hi = np.asarray(hi, dtype=np.float32) if hi is not None else None
    return policy, cfg, lo, hi, ckpt


def reset_arm(model, data):
    data.qpos[ARM_QPOS_START:ARM_QPOS_END] = np.array(
        [0.0, -0.5, 0.95, np.pi / 2.0, np.pi / 2.0], dtype=np.float64
    )
    data.qpos[GRIPPER_QPOS_START:GRIPPER_QPOS_END] = 0.3
    data.qvel[ARM_QPOS_START:GRIPPER_QPOS_END] = 0.0
    data.ctrl[ARM_QPOS_START:ARM_QPOS_END] = data.qpos[ARM_QPOS_START:ARM_QPOS_END]
    data.ctrl[GRIPPER_QPOS_START] = 0.3
    mujoco.mj_forward(model, data)


def reset_cube(data, cx: float, cy: float):
    data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [cx, cy, CUBE_REST_Z]
    data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[CUBE_QPOS_START:CUBE_QPOS_END] = 0.0
    mujoco.mj_forward(data.model, data)


def build_proprio(model, data, state_dim: int) -> np.ndarray:
    base = data.qpos[ARM_QPOS_START:GRIPPER_QPOS_END].astype(np.float32).copy()
    if state_dim == 6:
        return base
    is_g = is_grasping_from_contacts(model, data)
    if state_dim == 7:
        return np.concatenate([base, [is_g]]).astype(np.float32)
    if state_dim == 9:
        gripperframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
        target_rel = data.xpos[target_body_id, :2] - data.site_xpos[gripperframe_id, :2]
        return np.concatenate([base, target_rel, [is_g]]).astype(np.float32)
    raise ValueError(f"unsupported state_dim={state_dim}")


def stack_history(frame: np.ndarray, n_history_frames: int) -> np.ndarray:
    if n_history_frames <= 1:
        return frame
    window = np.repeat(frame[np.newaxis], n_history_frames, axis=0)
    return np.ascontiguousarray(
        np.transpose(window, (1, 2, 0, 3)).reshape(
            frame.shape[0], frame.shape[1], n_history_frames * frame.shape[-1]
        )
    )


def build_batch(image: np.ndarray, proprio: np.ndarray, text_ids: np.ndarray,
                cfg: BUDEConfig, device: str) -> tuple[dict, np.ndarray]:
    perception = detect_red_centroid(image, n_history_frames=cfg.n_history_frames)
    img_t = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    return {
        "images": img_t.to(device),
        "text_ids": torch.from_numpy(text_ids).unsqueeze(0).to(device),
        "instruction": [INSTRUCTION],
        "proprio": torch.from_numpy(proprio.astype(np.float32)).unsqueeze(0).to(device),
        "perception": torch.from_numpy(perception).unsqueeze(0).to(device),
        "domain_id": torch.tensor([_domain_from_instruction(INSTRUCTION)], dtype=torch.long).to(device),
    }, perception


def parse_xy(values: list[str]) -> list[tuple[float, float]]:
    if not values:
        return [(0.16, -0.09), (0.25, 0.0), (0.34, 0.09)]
    out = []
    for v in values:
        x, y = v.split(",", 1)
        out.append((float(x), float(y)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cube", action="append", default=[],
                    help="Cube xy as x,y. May be repeated. Defaults to three positions.")
    ap.add_argument("--raw", action="store_true", help="Print normalized actions instead of denormalized controls.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, cfg, lo, hi, ckpt = load_policy(args.ckpt, device)
    print(
        f"checkpoint step={ckpt.get('step')} img={cfg.img_size} chunk={cfg.chunk_size} "
        f"history={cfg.n_history_frames} state={cfg.state_dim} "
        f"bc={cfg.use_bc_head} context={cfg.use_context_action_head} "
        f"perception={cfg.use_perception}"
    )

    model = load_arm_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=cfg.img_size, width=cfg.img_size)
    front_top_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_top")
    wrist_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    text_ids = _tokenize_instruction(INSTRUCTION)

    rows = []
    with torch.no_grad():
        for cx, cy in parse_xy(args.cube):
            mujoco.mj_resetData(model, data)
            reset_arm(model, data)
            reset_cube(data, cx, cy)
            for _ in range(50):
                mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=front_top_cam)
            img_top = np.asarray(renderer.render()).copy()
            renderer.update_scene(data, camera=wrist_cam)
            img_wrist = np.asarray(renderer.render()).copy()
            image = stack_history(np.concatenate([img_top, img_wrist], axis=-1), cfg.n_history_frames)
            proprio = build_proprio(model, data, cfg.state_dim)
            batch, perception = build_batch(image, proprio, text_ids, cfg, device)
            chunk_norm = policy.sample(batch)[0].detach().cpu().numpy()
            first = chunk_norm[0] if args.raw or lo is None else denormalize_actions(chunk_norm[:1], lo, hi)[0]
            rows.append(first)
            print(
                f"cube=({cx:+.3f},{cy:+.3f}) perception=[{perception[0]:+.3f},{perception[1]:+.3f},{perception[2]:.0f}] "
                f"first_action={np.array2string(first, precision=4, suppress_small=False)}"
            )

    if len(rows) >= 2:
        rows = np.stack(rows, axis=0)
        span = rows.max(axis=0) - rows.min(axis=0)
        print(f"action_span={np.array2string(span, precision=5)} max_span={float(span.max()):.6f}")
        if float(span.max()) < 0.01:
            print("WARNING: first actions are still nearly identical across cube positions.")


if __name__ == "__main__":
    main()
