from __future__ import annotations
import mujoco
import numpy as np
from pathlib import Path

from bude_vla.envs.so101_mjx import (
    ARM_QPOS_START, ARM_QPOS_END,
    GRIPPER_QPOS_START, GRIPPER_QPOS_END,
    CUBE_QPOS_START, CUBE_QPOS_END,
    load_arm_model,
)
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace


INSTRUCTION = "pick up the red cube and place it in the blue target zone"


def record_pick_episode(root: str | Path, episode_idx: int = 0,
                        cube_xy: tuple[float, float] = (0.6, 0.0),
                        img_size: int = 64,
                        max_steps: int = 350) -> dict:
    model = load_arm_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [float(cube_xy[0]), float(cube_xy[1]), 0.445]
    data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=img_size, width=img_size)
    overhead_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                        "front_top")

    policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array(cube_xy))

    images, proprios, actions = [], [], []
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")

    for _ in range(max_steps):
        renderer.update_scene(data, camera=overhead_cam_id)
        img_overhead = np.asarray(renderer.render()).copy()
        images.append(img_overhead)
        arm_proprio = data.qpos[ARM_QPOS_START:GRIPPER_QPOS_END].astype(np.float32).copy()
        proprios.append(arm_proprio)

        ctrl, arm_target, done, info = policy.step(model, data)
        kinematic_action = np.concatenate([arm_target, [ctrl[GRIPPER_QPOS_START]]]).astype(np.float32)
        actions.append(kinematic_action)

        data.ctrl[:] = 0.0
        data.ctrl[GRIPPER_QPOS_START] = ctrl[GRIPPER_QPOS_START]
        data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0.0
        data.qpos[ARM_QPOS_START:ARM_QPOS_END] = arm_target
        policy._carry_cube_with(data)
        mujoco.mj_step(model, data)
        data.qpos[ARM_QPOS_START:ARM_QPOS_END] = arm_target
        policy._carry_cube_with(data)

        if done:
            break

    renderer.close()
    cube_final = data.xpos[cube_body_id].copy()
    target_pos = data.xpos[target_body_id].copy()
    success = bool(np.linalg.norm(cube_final[:2] - target_pos[:2]) < 0.10)

    return {
        "images": np.array(images, dtype=np.uint8),
        "proprio": np.array(proprios, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "instruction": INSTRUCTION,
        "success": success,
        "cube_final_xyz": cube_final,
        "target_xyz": target_pos,
    }
