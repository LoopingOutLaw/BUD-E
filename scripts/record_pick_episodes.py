"""Batch-record scripted pick-and-place episodes into LeRobot v3 layout.

Uses the v7 ggand0/pick-101 approach — physics-only grasping, no kinematic
carry/teleport. Only writes successful episodes to the training set.

Usage (headless, 100 episodes):
    unset PYTHONPATH
    MUJOCO_GL=egl PYTHONPATH=src python scripts/record_pick_episodes.py

Usage (smoke test, 5 episodes):
    MUJOCO_GL=egl PYTHONPATH=src python scripts/record_pick_episodes.py --max-eps 5
"""
from __future__ import annotations
import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from bude_vla.data.lerobot_v3 import write_episode, finalize_dataset
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace
from bude_vla.envs.so101_mjx import (
    load_arm_model,
    EXPERT_CONTROL_SUBSTEPS,
    PICK_WORKSPACE_X_RANGE,
    PICK_WORKSPACE_Y_RANGE,
    POLICY_RECORD_STRIDE,
    ARM_QPOS_START, ARM_QPOS_END,
    GRIPPER_QPOS_START,
    CUBE_QPOS_START,
    CUBE_REST_Z,
    is_grasping_from_contacts,
    is_cube_placed_in_bowl,
    build_pick_proprio,
)

INSTRUCTION = "pick up the red cube and place it in the blue target zone"



def _main_loop(
    model,
    data,
    policy,
    renderer,
    cam_ids,
    max_steps=2200,
    state_dim=10,
    record_stride=POLICY_RECORD_STRIDE,
    capture_observations=True,
):
    """Run the 125 Hz expert while recording a true ~31 Hz policy stream."""
    if record_stride < 1:
        raise ValueError("record_stride must be positive")
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")

    images, proprios, actions = [], [], []
    ever_grasped = False

    for step in range(max_steps):
        record_frame = step % record_stride == 0
        if record_frame and capture_observations:
            renderer.update_scene(data, camera=cam_ids[0])
            top = np.asarray(renderer.render()).copy()
            renderer.update_scene(data, camera=cam_ids[1])
            wrist = np.asarray(renderer.render()).copy()
            images.append(np.concatenate([top, wrist], axis=-1).copy())
            proprios.append(build_pick_proprio(model, data, state_dim))

        if is_grasping_from_contacts(model, data):
            ever_grasped = True

        ctrl, _arm_target, done, info = policy.step(model, data)
        data.ctrl[:] = ctrl
        for _ in range(EXPERT_CONTROL_SUBSTEPS):
            mujoco.mj_step(model, data)

        if info.get("attached"):
            ever_grasped = True

        if record_frame:
            action = np.concatenate([
                ctrl[ARM_QPOS_START:ARM_QPOS_END],
                [ctrl[GRIPPER_QPOS_START]],
            ]).astype(np.float32)
            actions.append(action)

        if done:
            break

    cube_final = data.xpos[cube_body_id].copy()
    target_pos = data.xpos[target_body_id].copy()
    reached_target = is_cube_placed_in_bowl(model, data)
    success = bool(reached_target and ever_grasped)
    sim_substeps = EXPERT_CONTROL_SUBSTEPS * record_stride
    fps = int(round(1.0 / (float(model.opt.timestep) * sim_substeps)))

    return {
        "images": np.array(images, dtype=np.uint8),
        "proprio": np.array(proprios, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "instruction": INSTRUCTION,
        "success": success,
        "ever_grasped": ever_grasped,
        "reached_target": reached_target,
        "cube_final_xyz": cube_final,
        "target_xyz": target_pos,
        "record_stride": int(record_stride),
        "sim_substeps_per_action": int(sim_substeps),
        "fps": fps,
    }


def _reset_pick_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_xy: tuple[float, float],
) -> None:
    """Reset to the exact state shared by recording and replay validation."""
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.0
    data.qpos[1] = -0.5
    data.qpos[2] = 0.95
    data.qpos[3] = np.pi / 2
    data.qpos[4] = np.pi / 2
    data.qpos[GRIPPER_QPOS_START] = 0.3
    data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [
        cube_xy[0], cube_xy[1], CUBE_REST_Z
    ]
    data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [
        1.0, 0.0, 0.0, 0.0
    ]
    data.qvel[:] = 0.0
    data.ctrl[:5] = data.qpos[:5]
    data.ctrl[GRIPPER_QPOS_START] = 0.3
    mujoco.mj_forward(model, data)
    for _ in range(50):
        mujoco.mj_step(model, data)


def _record_policy_rate_replay(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    cam_ids: tuple[int, int],
    cube_xy: tuple[float, float],
    actions: np.ndarray,
    state_dim: int,
    record_stride: int,
) -> dict:
    """Record observations while executing persisted actions at deployment rate.

    The high-rate IK expert supplies an action plan. This second pass is the
    trajectory used for learning, so every stored observation/action pair and
    every state transition exactly match policy rollout timing.
    """
    _reset_pick_state(model, data, cube_xy)
    cube_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "cube"
    )
    target_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_zone"
    )
    sim_substeps = EXPERT_CONTROL_SUBSTEPS * record_stride
    images: list[np.ndarray] = []
    proprios: list[np.ndarray] = []
    ever_grasped = False

    for action in actions:
        renderer.update_scene(data, camera=cam_ids[0])
        top = np.asarray(renderer.render()).copy()
        renderer.update_scene(data, camera=cam_ids[1])
        wrist = np.asarray(renderer.render()).copy()
        images.append(np.concatenate([top, wrist], axis=-1))
        proprios.append(build_pick_proprio(model, data, state_dim))

        data.ctrl[:] = np.asarray(action, dtype=np.float64)
        for _ in range(sim_substeps):
            mujoco.mj_step(model, data)
            ever_grasped = ever_grasped or bool(
                is_grasping_from_contacts(model, data) > 0.5
            )

    cube_final = data.xpos[cube_body_id].copy()
    target_pos = data.xpos[target_body_id].copy()
    reached_target = is_cube_placed_in_bowl(model, data)
    fps = int(round(1.0 / (float(model.opt.timestep) * sim_substeps)))
    return {
        "images": np.asarray(images, dtype=np.uint8),
        "proprio": np.asarray(proprios, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "instruction": INSTRUCTION,
        "success": bool(ever_grasped and reached_target),
        "ever_grasped": ever_grasped,
        "reached_target": reached_target,
        "cube_final_xyz": cube_final,
        "target_xyz": target_pos,
        "record_stride": int(record_stride),
        "sim_substeps_per_action": int(sim_substeps),
        "fps": fps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eps", type=int, default=500)
    ap.add_argument("--out", default="/home/aditya/bude_vla/data/pick_v10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-steps", type=int, default=2200)
    ap.add_argument("--record-stride", type=int, default=POLICY_RECORD_STRIDE,
                    help="Record every Nth 125 Hz expert step. Default 4 gives ~31 Hz.")
    ap.add_argument("--cube-x-range", nargs=2, type=float,
                    default=PICK_WORKSPACE_X_RANGE, metavar=("MIN", "MAX"),
                    help="Operational cube X range validated for top-down grasping.")
    ap.add_argument("--cube-y-range", nargs=2, type=float,
                    default=PICK_WORKSPACE_Y_RANGE, metavar=("MIN", "MAX"),
                    help="Operational cube Y range validated for top-down grasping.")
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--no-verify-replay", dest="verify_replay", action="store_false",
                    help="Record the decimated high-rate trajectory directly instead of "
                         "requiring and recording an exact policy-rate replay.")
    ap.add_argument("--state-dim", type=int, default=6, choices=[6, 7, 9, 10],
                    help="Recorded proprio dimension. Default 6 is deployable joint/gripper state only; "
                         "7/9/10 require additional robot-side or simulator signals.")
    ap.add_argument("--recovery-jitter-xy", type=float, default=0.0,
                    help="Max XY waypoint jitter in meters during approach/descent, decayed to zero before close.")
    ap.add_argument("--recovery-jitter-z", type=float, default=0.0,
                    help="Max Z/depth jitter in meters during descent, decayed to zero before close.")
    ap.add_argument("--recovery-jitter-prob", type=float, default=0.0,
                    help="Probability that an episode uses recoverable approach/descent jitter.")
    ap.add_argument("--max-grasp-retries", type=int, default=0,
                    help="Number of failed close attempts the scripted expert may recover from.")
    ap.add_argument("--nudge-recovery-prob", type=float, default=0.0,
                    help="Probability of a light touch/nudge during descent followed by backoff and clean retry.")
    ap.add_argument("--nudge-recovery-xy", type=float, default=0.0,
                    help="Max XY offset in meters for the induced descent nudge before recovery.")
    ap.add_argument("--nudge-recovery-z", type=float, default=0.0,
                    help="Max downward Z offset in meters for the induced descent nudge before recovery.")
    ap.add_argument("--retry-miss-xy", type=float, default=0.0,
                    help="Max XY close-target miss in meters on the first attempt for retry demos.")
    ap.add_argument("--retry-miss-prob", type=float, default=0.0,
                    help="Probability that an episode starts with an induced first-attempt miss.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = args.out
    os.makedirs(root, exist_ok=True)

    model = load_arm_model()
    camera_names = ["front_top", "wrist_cam"]
    cam_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cn)
                    for cn in camera_names)

    n_success = 0
    n_grasped_but_missed = 0
    n_never_grasped = 0
    n_replay_rejected = 0
    n_written = 0
    t0 = time.time()

    if (args.recovery_jitter_xy > 0.0 or args.recovery_jitter_z > 0.0) and args.recovery_jitter_prob > 0.0:
        print("  recovery jitter enabled: "
              f"xy=+/-{args.recovery_jitter_xy:.3f}m  "
              f"z=+/-{args.recovery_jitter_z:.3f}m  "
              f"prob={args.recovery_jitter_prob:.2f}")
    if args.max_grasp_retries > 0:
        print("  grasp retry demos enabled: "
              f"max_retries={args.max_grasp_retries}  "
              f"miss=+/-{args.retry_miss_xy:.3f}m  prob={args.retry_miss_prob:.2f}")
    if args.nudge_recovery_prob > 0.0:
        print("  nudge recovery demos enabled: "
              f"xy=+/-{args.nudge_recovery_xy:.3f}m  "
              f"z=-{args.nudge_recovery_z:.3f}m  "
              f"prob={args.nudge_recovery_prob:.2f}")
    if args.verify_replay:
        print("  exact policy-rate replay verification enabled: "
              "stored observations come from the 31.25 Hz replay")

    for i in range(args.max_eps):
        # Cube position: wide randomization to force visual grounding
        cx = float(rng.uniform(*args.cube_x_range))
        cy = float(rng.uniform(*args.cube_y_range))

        data = mujoco.MjData(model)
        _reset_pick_state(model, data, (cx, cy))

        policy = ScriptedPickAndPlace(
            model,
            data,
            cube_start_xy=np.array([cx, cy]),
            recovery_jitter_xy=args.recovery_jitter_xy,
            recovery_jitter_z=args.recovery_jitter_z,
            recovery_jitter_prob=args.recovery_jitter_prob,
            max_grasp_retries=args.max_grasp_retries,
            nudge_recovery_prob=args.nudge_recovery_prob,
            nudge_recovery_xy=args.nudge_recovery_xy,
            nudge_recovery_z=args.nudge_recovery_z,
            retry_miss_xy=args.retry_miss_xy,
            retry_miss_prob=args.retry_miss_prob,
            rng=rng,
        )
        renderer = mujoco.Renderer(model, height=args.img_size, width=args.img_size)

        ep = _main_loop(
            model,
            data,
            policy,
            renderer,
            cam_ids,
            max_steps=args.max_steps,
            state_dim=args.state_dim,
            record_stride=args.record_stride,
            capture_observations=(not args.verify_replay or args.keep_failures),
        )
        expert_success = bool(ep["success"])
        if args.verify_replay and expert_success:
            replay_data = mujoco.MjData(model)
            ep = _record_policy_rate_replay(
                model,
                replay_data,
                renderer,
                cam_ids,
                (cx, cy),
                ep["actions"],
                args.state_dim,
                args.record_stride,
            )
            if not ep["success"]:
                n_replay_rejected += 1

        ep["cube_start_xy"] = [cx, cy]
        renderer.close()

        if ep["success"]:
            n_success += 1
        elif ep["ever_grasped"]:
            n_grasped_but_missed += 1
        else:
            n_never_grasped += 1

        if ep["success"] or args.keep_failures:
            write_episode(root, ep)
            n_written += 1

        replay_status = (
            f"  replay={ep['success']}"
            if args.verify_replay and expert_success else ""
        )
        print(f"  ep {i:03d}  cube=({cx:.3f},{cy:.3f})  "
              f"steps={len(ep['actions'])}  success={ep['success']}  "
              f"grasped={ep['ever_grasped']}  reached={ep['reached_target']}  "
              f"expert={expert_success}{replay_status}"
              f"{'  [written]' if (ep['success'] or args.keep_failures) else '  [skipped]'}")

    elapsed = time.time() - t0
    rate = n_success / args.max_eps * 100 if args.max_eps > 0 else 0
    print(f"\n=== DONE  {n_success}/{args.max_eps} success ({rate:.0f}%)  "
          f"in {elapsed:.0f}s  out={root} ===")
    print(f"  never grasped: {n_never_grasped}")
    print(f"  grasped but missed target: {n_grasped_but_missed}")
    print(f"  expert successes rejected by policy-rate replay: {n_replay_rejected}")
    print(f"  episodes written: {n_written}")

    if n_written == 0:
        print("  No episodes written — nothing to finalize.")
        return

    stats = finalize_dataset(root)
    print(f"action_normalization persisted: "
          f"lo[0:3]={stats['lo'][:3]} ... hi[0:3]={stats['hi'][:3]}")


if __name__ == "__main__":
    main()
