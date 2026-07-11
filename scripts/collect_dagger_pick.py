"""Collect DAgger correction data from policy-visited pick states.

The rollout policy controls the simulator. At each visited state, this script
records the current observation/proprio and labels it with an IK expert action.
The expert may use simulator state to label data, but cube coordinates are not
stored in policy observations.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from bude_vla.action_space import (
    clip_arm_joint_targets,
    clip_gripper_control,
)
from bude_vla.data.action_normalization import denormalize_actions
from bude_vla.data.lerobot_v3 import finalize_dataset, write_episode
from bude_vla.data.lerobot_v3 import _tokenize_instruction
from bude_vla.data.lerobot_v3 import _domain_from_instruction
from bude_vla.envs.so101_mjx import (
    ARM_QPOS_START,
    ARM_QPOS_END,
    GRIPPER_QPOS_START,
    GRIPPER_QPOS_END,
    CUBE_QPOS_START,
    CUBE_QPOS_END,
    CUBE_REST_Z,
    N_ARM_JOINTS,
    PICK_WORKSPACE_X_RANGE,
    PICK_WORKSPACE_Y_RANGE,
    POLICY_CONTROL_SUBSTEPS,
    build_pick_proprio,
    is_grasping_from_contacts,
    is_touching_cube_from_contacts,
    load_arm_model,
)
from bude_vla.ik import IKController
from bude_vla.perception import detect_red_centroid
from bude_vla.scripted_pick_and_place import (
    ScriptedPickAndPlace,
    CLOSE,
    DESCENT,
    FINGER_WIDTH_OFFSET,
    GRASP_Z_OFFSET,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HEIGHT_OFFSET,
    WRIST_FLEX_LOCK,
    WRIST_ROLL_LOCK,
)
from eval_pick_ball import INSTRUCTION, load_policy, parse_cube_positions

SUBSTEPS_PER_FRAME = POLICY_CONTROL_SUBSTEPS
SUCCESS_THRESHOLD = 0.05


def reset_arm(model, data) -> None:
    data.qpos[0] = 0.0
    data.qpos[1] = -0.5
    data.qpos[2] = 0.95
    data.qpos[3] = WRIST_FLEX_LOCK
    data.qpos[4] = WRIST_ROLL_LOCK
    data.qpos[GRIPPER_QPOS_START] = GRIPPER_OPEN
    data.ctrl[:5] = data.qpos[:5]
    data.ctrl[GRIPPER_QPOS_START] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)


def reset_cube(data, cx: float, cy: float) -> None:
    data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [cx, cy, CUBE_REST_Z]
    data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[CUBE_QPOS_START:CUBE_QPOS_END] = 0.0
    mujoco.mj_forward(data.model, data)


def is_success(model, data) -> bool:
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    return bool(np.linalg.norm(data.xpos[cube_id, :2] - data.xpos[target_id, :2]) < SUCCESS_THRESHOLD)


def is_failure(model, data, step: int, max_steps: int) -> bool:
    if step >= max_steps:
        return True
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_pos = data.xpos[cube_id]
    if np.any(np.isnan(cube_pos)):
        return True
    if cube_pos[2] < -0.05 or cube_pos[2] > 1.5:
        return True
    if np.any(np.abs(data.qpos[ARM_QPOS_START:ARM_QPOS_END]) > 3.5):
        return True
    return False


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


def build_batch(image: np.ndarray, proprio: np.ndarray, text_ids: np.ndarray,
                device: str, n_history_frames: int) -> dict:
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


class DaggerPickExpert:
    """IK correction expert for policy-visited pick states."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ik = IKController(model, data, end_effector_site="gripperframe")
        self.cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
        self.gripper_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    def action(self) -> np.ndarray:
        cube = self.data.xpos[self.cube_id].copy()
        target = self.data.xpos[self.target_id].copy()
        ee = self.data.site_xpos[self.gripper_site].copy()
        grasping = is_grasping_from_contacts(self.model, self.data) > 0.5
        touching = is_touching_cube_from_contacts(self.model, self.data) > 0.5
        locked_joints = [3, 4]

        if grasping:
            if cube[2] < CUBE_REST_Z + 0.045:
                target_pos = cube.copy()
                target_pos[2] += HEIGHT_OFFSET + 0.06
                target_pos[1] += FINGER_WIDTH_OFFSET
                gain = 0.35
            elif np.linalg.norm(cube[:2] - target[:2]) > 0.04:
                target_pos = np.array([
                    target[0],
                    target[1] + FINGER_WIDTH_OFFSET,
                    CUBE_REST_Z + HEIGHT_OFFSET + 0.07,
                ])
                gain = 0.35
            else:
                target_pos = target.copy()
                target_pos[2] += GRASP_Z_OFFSET + 0.01
                target_pos[1] += FINGER_WIDTH_OFFSET
                gain = 0.25
            gripper_action = GRIPPER_CLOSED
        elif touching:
            target_pos = cube.copy()
            target_pos[2] += GRASP_Z_OFFSET
            target_pos[1] += FINGER_WIDTH_OFFSET
            gain = 0.30
            gripper_action = GRIPPER_CLOSED
        else:
            grasp_xy = cube[:2].copy()
            grasp_xy[1] += FINGER_WIDTH_OFFSET
            xy_err = np.linalg.norm(ee[:2] - grasp_xy)
            target_pos = cube.copy()
            target_pos[1] += FINGER_WIDTH_OFFSET
            if xy_err > 0.025 or ee[2] < cube[2] + GRASP_Z_OFFSET + 0.025:
                target_pos[2] += GRASP_Z_OFFSET + HEIGHT_OFFSET
                gain = 0.70
            else:
                target_pos[2] += GRASP_Z_OFFSET
                gain = 0.55
            gripper_action = GRIPPER_OPEN

        ctrl = self.ik.step_toward_target(
            target_pos,
            gripper_action=gripper_action,
            gain=gain,
            locked_joints=locked_joints,
        )
        ctrl[3] = WRIST_FLEX_LOCK
        ctrl[4] = WRIST_ROLL_LOCK
        return np.concatenate([ctrl[:N_ARM_JOINTS], [ctrl[GRIPPER_QPOS_START]]]).astype(np.float32)


def next_policy_action(policy, batch, cfg, action_lo, action_hi, *,
                       exec_first_only: bool, chunk_state: dict) -> np.ndarray:
    if exec_first_only:
        chunk = policy.sample(batch)[0].detach().cpu().numpy()
        action = chunk[0]
    else:
        if chunk_state.get("chunk") is None or chunk_state.get("cursor", 0) >= cfg.chunk_size:
            chunk_state["chunk"] = policy.sample(batch)[0].detach().cpu().numpy()
            chunk_state["cursor"] = 0
        action = chunk_state["chunk"][chunk_state["cursor"]]
        chunk_state["cursor"] += 1
    if action_lo is not None:
        action = denormalize_actions(action, action_lo, action_hi)
    return action


def rollout_episode(model, policy, cfg, action_lo, action_hi, device: str,
                    obs_renderer, cube_xy: tuple[float, float], *,
                    max_steps: int, record_state_dim: int,
                    exec_first_only: bool, intervention_mode: bool,
                    intervention_steps: int, near_trigger_dist: float,
                    intervention_trigger: str, max_interventions: int,
                    allow_retrigger: bool) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    reset_arm(model, data)
    reset_cube(data, cube_xy[0], cube_xy[1])
    for _ in range(50):
        mujoco.mj_step(model, data)

    front_top_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_top")
    wrist_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    text_ids = _tokenize_instruction(INSTRUCTION)
    correction_expert = DaggerPickExpert(model, data)
    scripted_expert: ScriptedPickAndPlace | None = None
    scripted_done = False
    img_buffer: list[np.ndarray] = []
    chunk_state: dict = {"chunk": None, "cursor": 0}

    images: list[np.ndarray] = []
    proprios: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    ever_grasped = False
    intervention_until = -1
    intervention_started = False
    intervention_frames = 0
    interventions_started = 0
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    gripper_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    for step in range(max_steps):
        obs_renderer.update_scene(data, camera=front_top_cam)
        top = np.asarray(obs_renderer.render()).copy()
        obs_renderer.update_scene(data, camera=wrist_cam)
        wrist = np.asarray(obs_renderer.render()).copy()
        image = np.concatenate([top, wrist], axis=-1)
        img_buffer.append(image)
        stacked = stack_observation(img_buffer, cfg.n_history_frames)

        record_proprio = build_pick_proprio(model, data, record_state_dim)
        policy_proprio = build_pick_proprio(model, data, cfg.state_dim)

        if is_grasping_from_contacts(model, data) > 0.5:
            ever_grasped = True

        batch = build_batch(stacked, policy_proprio, text_ids, device, cfg.n_history_frames)
        touching = is_touching_cube_from_contacts(model, data) > 0.5
        cube = data.xpos[cube_id].copy()
        ee = data.site_xpos[gripper_site].copy()
        near_cube = (np.linalg.norm(ee[:2] - cube[:2]) <= near_trigger_dist
                     and abs(float(ee[2] - cube[2])) <= 0.10)
        if intervention_trigger == "touch":
            trigger_now = touching
        elif intervention_trigger == "near":
            trigger_now = near_cube
        else:
            trigger_now = touching or near_cube
        can_start_intervention = (
            intervention_mode
            and trigger_now
            and step < max_steps - 5
            and interventions_started < max(1, max_interventions)
            and (allow_retrigger or interventions_started == 0 or step > intervention_until)
        )
        if can_start_intervention:
            intervention_started = True
            interventions_started += 1
            intervention_until = step + intervention_steps
            scripted_done = False
            scripted_expert = ScriptedPickAndPlace(
                model, data, cube[:2], max_grasp_retries=1,
                recovery_jitter_xy=0.0, recovery_jitter_z=0.0,
                nudge_recovery_prob=0.0,
            )
            # Short DAgger interventions should teach the local correction, not
            # replay the whole scripted trajectory from APPROACH.
            scripted_expert.phase = CLOSE if touching else DESCENT
            scripted_expert.phase_step = 0

        if intervention_mode and step <= intervention_until and scripted_expert is not None and not scripted_done:
            ctrl, _arm_q, scripted_done, _info = scripted_expert.step(model, data)
            expert_action = np.concatenate([
                ctrl[:N_ARM_JOINTS], [ctrl[GRIPPER_QPOS_START]]
            ]).astype(np.float32)
        else:
            expert_action = correction_expert.action()

        images.append(image.copy())
        proprios.append(record_proprio)
        expert_actions.append(expert_action)

        with torch.no_grad():
            policy_action = next_policy_action(
                policy, batch, cfg, action_lo, action_hi,
                exec_first_only=exec_first_only,
                chunk_state=chunk_state,
            )

        if np.any(np.isnan(policy_action)):
            break
        if intervention_mode and step <= intervention_until and scripted_expert is not None and not scripted_done:
            exec_action = expert_action
            intervention_frames += 1
        else:
            exec_action = policy_action
        arm_target = clip_arm_joint_targets(model, exec_action)
        gripper_ctrl = clip_gripper_control(model, float(exec_action[N_ARM_JOINTS]))
        data.ctrl[:N_ARM_JOINTS] = arm_target
        data.ctrl[GRIPPER_QPOS_START] = gripper_ctrl
        for _ in range(SUBSTEPS_PER_FRAME):
            mujoco.mj_step(model, data)

        if is_success(model, data):
            break
        if is_failure(model, data, step, max_steps):
            break

    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    cube_final = data.xpos[cube_id].copy()
    target_pos = data.xpos[target_id].copy()
    reached_target = bool(np.linalg.norm(cube_final[:2] - target_pos[:2]) < SUCCESS_THRESHOLD)
    proprio_arr = np.array(proprios, dtype=np.float32)
    any_contact_frames = int((proprio_arr[:, 8] > 0.5).sum()) if proprio_arr.ndim == 2 and proprio_arr.shape[1] >= 10 else 0
    strict_grasp_frames = int((proprio_arr[:, 9] > 0.5).sum()) if proprio_arr.ndim == 2 and proprio_arr.shape[1] >= 10 else 0
    return {
        "images": np.array(images, dtype=np.uint8),
        "proprio": proprio_arr,
        "actions": np.array(expert_actions, dtype=np.float32),
        "instruction": INSTRUCTION,
        "success": bool(reached_target and ever_grasped),
        "ever_grasped": ever_grasped,
        "reached_target": reached_target,
        "any_contact_frames": any_contact_frames,
        "strict_grasp_frames": strict_grasp_frames,
        "intervention_started": intervention_started,
        "intervention_frames": intervention_frames,
        "interventions_started": interventions_started,
        "cube_final_xyz": cube_final,
        "target_xyz": target_pos,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-episodes", type=int, default=200,
                    help="Target number of written episodes when --min-contact-frames > 0; otherwise rollout attempts.")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="Maximum rollout attempts for contact-filtered collection. Defaults to --num-episodes.")
    ap.add_argument("--min-contact-frames", type=int, default=0,
                    help="Only write episodes with at least this many any-contact frames. 0 keeps all non-short episodes.")
    ap.add_argument("--min-grasp-frames", type=int, default=0,
                    help="Only write episodes with at least this many strict two-pad grasp frames.")
    ap.add_argument("--require-success", action="store_true",
                    help="Only write episodes where the cube is both grasped and reaches the target zone.")
    ap.add_argument("--intervention-mode", action="store_true",
                    help="After policy near/contact trigger, execute expert actions for a recovery window while recording expert labels.")
    ap.add_argument("--intervention-steps", type=int, default=280,
                    help="Upper bound on expert intervention window after each trigger. CLOSE needs ~250+ steps to produce strict grasp.")
    ap.add_argument("--intervention-trigger", choices=["touch", "near", "both"], default="touch",
                    help="Trigger expert intervention on strict robot contact, near-cube proximity, or either.")
    ap.add_argument("--max-interventions", type=int, default=1,
                    help="Maximum expert intervention windows per episode.")
    ap.add_argument("--allow-retrigger", action="store_true",
                    help="Allow a new intervention after the previous window ends.")
    ap.add_argument("--near-trigger-dist", type=float, default=0.030,
                    help="XY gripper-to-cube distance that triggers near intervention, in meters.")
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--state-dim", type=int, default=10, choices=[9, 10],
                    help="Recorded DAgger proprio dimension. Use 10 for any-contact + grasp.")
    ap.add_argument("--exec-first-only", action="store_true",
                    help="Replan each step and execute only the first action, matching current eval debugging.")
    ap.add_argument("--cube-positions", default=None,
                    help="Optional fixed x,y;x,y list. Otherwise samples the training cube range.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, action_lo, action_hi, cfg = load_policy(args.ckpt, args.img_size, device)
    model = load_arm_model()
    obs_renderer = mujoco.Renderer(model, height=cfg.img_size, width=cfg.img_size)
    rng = np.random.default_rng(args.seed)
    fixed_positions = parse_cube_positions(args.cube_positions)

    os.makedirs(args.out, exist_ok=True)
    target_written = int(args.num_episodes)
    max_attempts = int(args.max_attempts) if args.max_attempts is not None else target_written
    if args.min_contact_frames <= 0:
        max_attempts = target_written
    n_written = 0
    n_success = 0
    n_short = 0
    n_contact_poor = 0
    t0 = time.time()
    for attempt_i in range(max_attempts):
        if args.min_contact_frames > 0 and n_written >= target_written:
            break
        if fixed_positions:
            cube_xy = fixed_positions[attempt_i % len(fixed_positions)]
        else:
            cube_xy = (float(rng.uniform(*PICK_WORKSPACE_X_RANGE)), float(rng.uniform(*PICK_WORKSPACE_Y_RANGE)))
        ep = rollout_episode(
            model, policy, cfg, action_lo, action_hi, device, obs_renderer, cube_xy,
            max_steps=args.max_steps,
            record_state_dim=args.state_dim,
            exec_first_only=args.exec_first_only,
            intervention_mode=args.intervention_mode,
            intervention_steps=args.intervention_steps,
            near_trigger_dist=args.near_trigger_dist,
            intervention_trigger=args.intervention_trigger,
            max_interventions=args.max_interventions,
            allow_retrigger=args.allow_retrigger,
        )
        long_enough = len(ep["actions"]) >= max(2, cfg.chunk_size)
        contact_enough = int(ep["any_contact_frames"]) >= max(0, args.min_contact_frames)
        grasp_enough = int(ep["strict_grasp_frames"]) >= max(0, args.min_grasp_frames)
        success_enough = (not args.require_success) or bool(ep["success"])
        intervention_enough = (not args.intervention_mode) or bool(ep["intervention_started"])
        should_write = long_enough and contact_enough and grasp_enough and success_enough and intervention_enough
        if should_write:
            write_episode(args.out, ep)
            n_written += 1
            n_success += int(ep["success"])
        elif not long_enough:
            n_short += 1
        else:
            n_contact_poor += 1
        if should_write:
            status = "[written]"
        elif not long_enough:
            status = "[skipped-short]"
        elif not intervention_enough:
            status = "[skipped-no-intervention]"
        elif not grasp_enough:
            status = "[skipped-grasp]"
        elif not success_enough:
            status = "[skipped-success]"
        else:
            status = "[skipped-contact]"
        print(
            f"  attempt {attempt_i:04d} written={n_written:04d}/{target_written:04d} "
            f"cube=({cube_xy[0]:.3f},{cube_xy[1]:.3f}) "
            f"steps={len(ep['actions'])} contact={ep['any_contact_frames']} "
            f"grasp_frames={ep['strict_grasp_frames']} intervention={ep['intervention_frames']} "
            f"n_interventions={ep['interventions_started']} "
            f"success={ep['success']} "
            f"grasped={ep['ever_grasped']} reached={ep['reached_target']} {status}",
            flush=True,
        )

    obs_renderer.close()
    if n_written:
        finalize_dataset(args.out)
    elapsed = time.time() - t0
    print(
        f"\n=== DAGGER DONE wrote={n_written}/{target_written} attempts={attempt_i + 1 if max_attempts else 0}/{max_attempts} "
        f"success={n_success} skipped_short={n_short} skipped_contact={n_contact_poor} "
        f"in {elapsed:.0f}s out={args.out} ==="
    )


if __name__ == "__main__":
    main()
