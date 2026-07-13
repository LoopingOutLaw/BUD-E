"""Fast random-position benchmark for pick policies.

Unlike eval_pick_ball.py, this script does not write video. It measures broad
random-position rollout health: any-pad contact, strict two-pad grasp, and final
success. Use it to catch narrow overfitting to a tiny fixed eval set.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from bude_vla.action_space import apply_policy_action, make_ik_controller, uses_ik_action_space
from bude_vla.data.action_normalization import denormalize_actions
from bude_vla.data.lerobot_v3 import _tokenize_instruction
from bude_vla.envs.so101_mjx import (
    GRIPPER_QPOS_START,
    PICK_WORKSPACE_X_RANGE,
    PICK_WORKSPACE_Y_RANGE,
    N_ARM_JOINTS,
    build_pick_proprio,
    is_grasping_from_contacts,
    is_touching_cube_from_contacts,
    load_arm_model,
)
from bude_vla.perception import detect_red_centroid
from eval_pick_ball import (
    INSTRUCTION,
    MAX_STEPS,
    SUBSTEPS_PER_FRAME,
    build_batch,
    is_failure,
    is_success,
    load_policy,
    parse_cube_positions,
    reset_arm,
    reset_cube,
)


def stack_observation(buffer: list[np.ndarray], n_history_frames: int) -> np.ndarray:
    if n_history_frames <= 1:
        return buffer[-1]
    while len(buffer) < n_history_frames:
        buffer.insert(0, buffer[0])
    window = np.stack(buffer[-n_history_frames:], axis=0)
    window = np.ascontiguousarray(window)
    return np.transpose(window, (1, 2, 0, 3)).reshape(
        window.shape[1], window.shape[2], n_history_frames * window.shape[-1]
    )


def sample_action(policy, batch, cfg, action_lo, action_hi, *,
                  exec_first_only: bool, ensembling: bool,
                  ensembling_k: float, replan_every: int,
                  execute_horizon: int,
                  state: dict, step: int) -> np.ndarray:
    if ensembling:
        action_queue = state.setdefault("action_queue", [])
        if not action_queue or step % max(1, replan_every) == 0:
            new_chunk = policy.sample(batch)[0].detach().cpu().numpy()
            if action_lo is not None:
                new_chunk = denormalize_actions(new_chunk, action_lo, action_hi)
            q = list(action_queue)
            for i, new_a in enumerate(new_chunk):
                if i < len(q):
                    q[i] = ensembling_k * q[i] + (1.0 - ensembling_k) * new_a
                else:
                    q.append(new_a)
            action_queue = q
        action = action_queue.pop(0)
        state["action_queue"] = action_queue
        return action

    if exec_first_only:
        chunk = policy.sample(batch)[0].detach().cpu().numpy()
        action = chunk[0]
        if action_lo is not None:
            action = denormalize_actions(action, action_lo, action_hi)
        return action

    horizon = cfg.chunk_size if execute_horizon <= 0 else min(
        cfg.chunk_size, execute_horizon
    )
    if state.get("chunk") is None or state.get("cursor", 0) >= horizon:
        state["chunk"] = policy.sample(batch)[0].detach().cpu().numpy()
        state["cursor"] = 0
    action = state["chunk"][state["cursor"]]
    state["cursor"] += 1
    if action_lo is not None:
        action = denormalize_actions(action, action_lo, action_hi)
    return action


def run_one(policy, model, data, renderer, text_ids, action_lo, action_hi, cfg,
            device: str, cube_xy: tuple[float, float], *, max_steps: int,
            exec_first_only: bool, ensembling: bool, ensembling_k: float,
            replan_every: int, execute_horizon: int,
            contact_close_reflex: bool,
            contact_close_steps: int, contact_close_value: float) -> dict:
    front_top_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_top")
    wrist_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

    mujoco.mj_resetData(model, data)
    reset_arm(model, data)
    reset_cube(data, cube_xy[0], cube_xy[1])
    for _ in range(50):
        mujoco.mj_step(model, data)

    img_buffer: list[np.ndarray] = []
    action_state: dict = {"chunk": None, "cursor": 0, "action_queue": []}
    touch_frames = 0
    grasp_frames = 0
    first_touch_step: int | None = None
    first_grasp_step: int | None = None
    close_until = -1
    ik = make_ik_controller(model, data) if uses_ik_action_space(cfg) else None

    for step in range(max_steps):
        renderer.update_scene(data, camera=front_top_cam)
        top = np.asarray(renderer.render()).copy()
        renderer.update_scene(data, camera=wrist_cam)
        wrist = np.asarray(renderer.render()).copy()
        image = np.concatenate([top, wrist], axis=-1)
        img_buffer.append(image)
        stacked = stack_observation(img_buffer, cfg.n_history_frames)

        touching = is_touching_cube_from_contacts(model, data) > 0.5
        grasping = is_grasping_from_contacts(model, data) > 0.5
        if touching:
            touch_frames += 1
            if first_touch_step is None:
                first_touch_step = step
            if contact_close_reflex:
                close_until = max(close_until, step + contact_close_steps)
        if grasping:
            grasp_frames += 1
            if first_grasp_step is None:
                first_grasp_step = step

        proprio = build_pick_proprio(model, data, cfg.state_dim)
        batch = build_batch(stacked, proprio, text_ids, device, n_history_frames=cfg.n_history_frames)
        with torch.no_grad():
            action = sample_action(
                policy, batch, cfg, action_lo, action_hi,
                exec_first_only=exec_first_only,
                ensembling=ensembling,
                ensembling_k=ensembling_k,
                replan_every=replan_every,
                execute_horizon=execute_horizon,
                state=action_state,
                step=step,
            )

        if np.any(np.isnan(action)):
            break
        apply_policy_action(
            model, data, action, cfg, ik=ik,
            contact_close_reflex=contact_close_reflex,
            close_active=step <= close_until,
            contact_close_value=contact_close_value,
        )
        for _ in range(SUBSTEPS_PER_FRAME):
            mujoco.mj_step(model, data)

        if is_success(data) and grasp_frames > 0:
            return {
                "success": True,
                "touch_frames": touch_frames,
                "grasp_frames": grasp_frames,
                "first_touch_step": first_touch_step,
                "first_grasp_step": first_grasp_step,
                "steps": step + 1,
            }
        if is_failure(data, step, max_steps=max_steps):
            break

    return {
        "success": False,
        "touch_frames": touch_frames,
        "grasp_frames": grasp_frames,
        "first_touch_step": first_touch_step,
        "first_grasp_step": first_grasp_step,
        "steps": max_steps,
    }


def print_workspace_breakdown(
    results: list[dict],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    bins: int = 4,
) -> None:
    """Print spatial and stage failures without exposing them to the policy."""
    no_touch = sum(r["touch_frames"] == 0 for r in results)
    touch_no_grasp = sum(
        r["touch_frames"] > 0 and r["grasp_frames"] == 0 for r in results
    )
    grasp_no_success = sum(
        r["grasp_frames"] > 0 and not r["success"] for r in results
    )
    print(
        "failure stages: "
        f"no_contact={no_touch} touch_no_grasp={touch_no_grasp} "
        f"grasp_no_success={grasp_no_success}"
    )

    for axis, value_index, limits in (
        ("x", 0, x_range),
        ("y", 1, y_range),
    ):
        edges = np.linspace(limits[0], limits[1], bins + 1)
        print(f"{axis}-workspace bins:")
        for i in range(bins):
            include_hi = i == bins - 1
            selected = [
                r for r in results
                if r["cube_xy"][value_index] >= edges[i]
                and (
                    r["cube_xy"][value_index] <= edges[i + 1]
                    if include_hi else
                    r["cube_xy"][value_index] < edges[i + 1]
                )
            ]
            n_bin = len(selected)
            if n_bin == 0:
                continue
            contact = sum(r["touch_frames"] > 0 for r in selected) / n_bin
            grasp = sum(r["grasp_frames"] > 0 for r in selected) / n_bin
            success = sum(bool(r["success"]) for r in selected) / n_bin
            closing = "]" if include_hi else ")"
            print(
                f"  [{edges[i]:+.4f},{edges[i + 1]:+.4f}{closing} "
                f"n={n_bin} contact={contact:.3f} grasp={grasp:.3f} "
                f"success={success:.3f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--raw-weights", action="store_true",
                    help="Evaluate model_state_dict instead of EMA weights.")
    ap.add_argument("--num-episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--cube-positions", default=None)
    ap.add_argument("--cube-x-range",
                    default=",".join(str(v) for v in PICK_WORKSPACE_X_RANGE))
    ap.add_argument("--cube-y-range",
                    default=",".join(str(v) for v in PICK_WORKSPACE_Y_RANGE))
    ap.add_argument("--exec-first-only", action="store_true")
    ap.add_argument("--ensembling", action="store_true")
    ap.add_argument("--ensembling-k", type=float, default=0.55)
    ap.add_argument("--replan-every", type=int, default=1)
    ap.add_argument(
        "--execute-horizon", type=int, default=0,
        help=("For non-ensembled chunk execution, replan after this many "
              "actions; 0 executes the full trained chunk."),
    )
    ap.add_argument("--contact-close-reflex", action="store_true",
                    help="Robot-side reflex: close/hold gripper briefly after any-pad cube contact.")
    ap.add_argument("--contact-close-steps", type=int, default=120)
    ap.add_argument("--contact-close-value", type=float, default=-1.0)
    ap.add_argument("--min-success-rate", type=float, default=0.0,
                    help="Exit nonzero when final success is below this fraction.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, action_lo, action_hi, cfg = load_policy(
        args.ckpt, args.img_size, device, use_ema=not args.raw_weights
    )
    model = load_arm_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=cfg.img_size, width=cfg.img_size)
    text_ids = _tokenize_instruction(INSTRUCTION)
    rng = np.random.default_rng(args.seed)
    fixed_positions = parse_cube_positions(args.cube_positions)
    x_range = tuple(float(v) for v in args.cube_x_range.split(","))
    y_range = tuple(float(v) for v in args.cube_y_range.split(","))

    results = []
    for ep in range(args.num_episodes):
        torch.manual_seed(args.seed + ep)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + ep)
        if fixed_positions:
            cube_xy = fixed_positions[ep % len(fixed_positions)]
        else:
            cube_xy = (float(rng.uniform(*x_range)), float(rng.uniform(*y_range)))
        result = run_one(
            policy, model, data, renderer, text_ids, action_lo, action_hi, cfg, device, cube_xy,
            max_steps=args.max_steps,
            exec_first_only=args.exec_first_only,
            ensembling=args.ensembling,
            ensembling_k=args.ensembling_k,
            replan_every=args.replan_every,
            execute_horizon=args.execute_horizon,
            contact_close_reflex=args.contact_close_reflex,
            contact_close_steps=args.contact_close_steps,
            contact_close_value=args.contact_close_value,
        )
        result["cube_xy"] = cube_xy
        results.append(result)
        print(
            f"ep {ep:03d} cube=({cube_xy[0]:.3f},{cube_xy[1]:.3f}) "
            f"touch={result['touch_frames']} grasp={result['grasp_frames']} "
            f"success={result['success']} steps={result['steps']}",
            flush=True,
        )

    renderer.close()
    n = len(results)
    successes = sum(int(r["success"]) for r in results)
    touched = sum(int(r["touch_frames"] > 0) for r in results)
    grasped = sum(int(r["grasp_frames"] > 0) for r in results)
    touch_frames = sum(int(r["touch_frames"]) for r in results)
    grasp_frames = sum(int(r["grasp_frames"]) for r in results)
    first_touch = [r["first_touch_step"] for r in results if r["first_touch_step"] is not None]

    print("\n=== RANDOM PICK BENCH ===")
    print(f"checkpoint: {args.ckpt}")
    print(f"episodes: {n}")
    print(f"success episodes: {successes}/{n} ({successes / n:.3f})")
    print(f"any_contact episodes: {touched}/{n} ({touched / n:.3f})")
    print(f"strict_grasp episodes: {grasped}/{n} ({grasped / n:.3f})")
    print(f"any_contact frames: {touch_frames}")
    print(f"strict_grasp frames: {grasp_frames}")
    print(f"avg any_contact frames/episode: {touch_frames / n:.2f}")
    print(f"avg strict_grasp frames/episode: {grasp_frames / n:.2f}")
    if first_touch:
        print(f"median first_touch_step: {float(np.median(first_touch)):.1f}")
    else:
        print("median first_touch_step: NA")
    print_workspace_breakdown(results, x_range, y_range)

    success_rate = successes / max(1, n)
    if success_rate < args.min_success_rate:
        raise SystemExit(
            f"success {success_rate:.3f} is below required "
            f"{args.min_success_rate:.3f}"
        )


if __name__ == "__main__":
    main()
