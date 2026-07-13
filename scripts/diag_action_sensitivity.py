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

from bude_vla.action_space import joint_action_to_ee_abs
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
    EXPERT_CONTROL_SUBSTEPS,
    POLICY_RECORD_STRIDE,
    load_arm_model,
    is_grasping_from_contacts,
    build_pick_proprio,
)
from bude_vla.models.policy import BUDEConfig, BUDEPolicy, apply_saved_config
from bude_vla.perception import detect_red_centroid
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace

INSTRUCTION = "pick up the red cube and place it in the blue target zone"


def load_policy(path: str, device: str, *, use_ema: bool = True):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved = ckpt.get("config", {})
    cfg = BUDEConfig()
    cfg.chunk_size = 4
    apply_saved_config(cfg, saved)
    cfg.patch_size = 16

    policy = BUDEPolicy(cfg).to(device)
    state_dict = (
        ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
        if use_ema else ckpt["model_state_dict"]
    )
    policy.load_state_dict(state_dict)
    policy.eval()

    lo = ckpt.get("action_norm_lo")
    hi = ckpt.get("action_norm_hi")
    lo = np.asarray(lo, dtype=np.float32) if lo is not None else None
    hi = np.asarray(hi, dtype=np.float32) if hi is not None else None
    return policy, cfg, lo, hi, ckpt


def expert_initial_chunk(model, data, cube_xy, chunk_size: int) -> np.ndarray:
    """Generate the same policy-rate expert controls used by the recorder."""
    expert = ScriptedPickAndPlace(
        model, data, np.asarray(cube_xy, dtype=np.float64)
    )
    fk_data = mujoco.MjData(model)
    actions = []
    max_expert_steps = (chunk_size - 1) * POLICY_RECORD_STRIDE + 1
    for expert_step in range(max_expert_steps):
        ctrl, _arm_q, _done, _info = expert.step(model, data)
        if expert_step % POLICY_RECORD_STRIDE == 0:
            joint_action = np.concatenate([ctrl[:5], [ctrl[5]]])
            actions.append(joint_action_to_ee_abs(
                model, fk_data, joint_action
            ))
        data.ctrl[:] = ctrl
        for _ in range(EXPERT_CONTROL_SUBSTEPS):
            mujoco.mj_step(model, data)
    return np.asarray(actions, dtype=np.float32)


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
    return build_pick_proprio(model, data, state_dim)


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
    ap.add_argument("--raw", "--normalized-actions", dest="normalized_actions",
                    action="store_true",
                    help="Print normalized actions instead of denormalized controls.")
    ap.add_argument("--raw-weights", action="store_true",
                    help="Evaluate model_state_dict instead of EMA weights.")
    ap.add_argument("--min-shoulder-span", type=float, default=0.0,
                    help="Exit nonzero unless denormalized shoulder-pan span reaches this value.")
    ap.add_argument("--min-shoulder-lift-span", type=float, default=0.0,
                    help="Exit nonzero unless denormalized shoulder-lift span reaches this value.")
    ap.add_argument("--max-task-space-p95-mm", type=float, default=0.0,
                    help="Exit nonzero if chunk-endpoint TCP p95 error exceeds this value.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, cfg, lo, hi, ckpt = load_policy(
        args.ckpt, device, use_ema=not args.raw_weights
    )
    print(
        f"checkpoint step={ckpt.get('step')} img={cfg.img_size} chunk={cfg.chunk_size} "
        f"history={cfg.n_history_frames} state={cfg.state_dim} "
        f"bc={cfg.use_bc_head} context={cfg.use_context_action_head} "
        f"perception={cfg.use_perception} "
        f"weights={'raw' if args.raw_weights else 'ema'}"
    )

    model = load_arm_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=cfg.img_size, width=cfg.img_size)
    front_top_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_top")
    wrist_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    text_ids = _tokenize_instruction(INSTRUCTION)

    first_rows = []
    endpoint_rows = []
    endpoint_errors_m = []
    cube_rows = []
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
            chunk = (
                chunk_norm if args.normalized_actions or lo is None
                else denormalize_actions(chunk_norm, lo, hi)
            )
            first = chunk[0]
            endpoint = chunk[-1]
            first_rows.append(first)
            endpoint_rows.append(endpoint)
            cube_rows.append([cx, cy])
            expert_endpoint = None
            if not args.normalized_actions and lo is not None:
                expert_chunk = expert_initial_chunk(
                    model, data, (cx, cy), cfg.chunk_size
                )
                expert_endpoint = expert_chunk[-1]
                endpoint_errors_m.append(float(np.linalg.norm(
                    endpoint[:3] - expert_endpoint[:3]
                )))
            print(
                f"cube=({cx:+.3f},{cy:+.3f}) perception=[{perception[0]:+.3f},{perception[1]:+.3f},{perception[2]:.0f}] "
                f"first={np.array2string(first, precision=4, suppress_small=False)} "
                f"endpoint={np.array2string(endpoint, precision=4, suppress_small=False)}"
            )
            if expert_endpoint is not None:
                print(
                    "  expert_endpoint="
                    f"{np.array2string(expert_endpoint, precision=4, suppress_small=False)}"
                )

    if len(first_rows) >= 2:
        first_rows = np.stack(first_rows, axis=0)
        endpoint_rows = np.stack(endpoint_rows, axis=0)
        cube_rows = np.asarray(cube_rows, dtype=np.float64)
        first_span = np.ptp(first_rows, axis=0)
        endpoint_span = np.ptp(endpoint_rows, axis=0)
        print(f"first_action_span={np.array2string(first_span, precision=5)}")
        print(f"chunk_endpoint_span={np.array2string(endpoint_span, precision=5)}")
        if cfg.action_space == "ee_abs" and not args.normalized_actions:
            for name, cube_dim, action_dim in (("x", 0, 0), ("y", 1, 1)):
                slope, _intercept = np.polyfit(
                    cube_rows[:, cube_dim], endpoint_rows[:, action_dim], 1
                )
                corr = np.corrcoef(
                    cube_rows[:, cube_dim], endpoint_rows[:, action_dim]
                )[0, 1]
                print(
                    f"chunk_endpoint_{name} slope={slope:.4f} corr={corr:.4f}"
                )
        shoulder_span = float(first_span[0])
        shoulder_lift_span = float(first_span[1]) if len(first_span) > 1 else 0.0
        task_p95_mm = (
            float(np.percentile(endpoint_errors_m, 95) * 1000.0)
            if endpoint_errors_m else float("nan")
        )
        if endpoint_errors_m:
            print(
                "chunk_endpoint_error_mm "
                f"median={np.median(endpoint_errors_m) * 1000.0:.3f} "
                f"p95={task_p95_mm:.3f} "
                f"max={np.max(endpoint_errors_m) * 1000.0:.3f}"
            )
        renderer.close()
        if args.min_shoulder_span > 0.0 and shoulder_span < args.min_shoulder_span:
            print(
                f"ERROR: shoulder-pan span {shoulder_span:.6f} is below "
                f"required {args.min_shoulder_span:.6f}"
            )
            raise SystemExit(2)
        if (
            args.min_shoulder_lift_span > 0.0
            and shoulder_lift_span < args.min_shoulder_lift_span
        ):
            print(
                f"ERROR: shoulder-lift span {shoulder_lift_span:.6f} is below "
                f"required {args.min_shoulder_lift_span:.6f}"
            )
            raise SystemExit(2)
        if (
            args.max_task_space_p95_mm > 0.0
            and (
                not np.isfinite(task_p95_mm)
                or task_p95_mm > args.max_task_space_p95_mm
            )
        ):
            print(
                f"ERROR: task-space p95 error {task_p95_mm:.3f}mm exceeds "
                f"required {args.max_task_space_p95_mm:.3f}mm"
            )
            raise SystemExit(2)
    else:
        renderer.close()


if __name__ == "__main__":
    main()
