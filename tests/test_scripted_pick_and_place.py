"""Tests for scripted pick-and-place policy."""
import mujoco
import numpy as np
from pathlib import Path
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace


MODEL_PATH = Path(__file__).resolve().parents[1] / "urdf" / "ur5e_scene.xml"


def _step_once(model, data, policy):
    ctrl, arm_target, done, info = policy.step(model, data)
    data.ctrl[:] = 0.0
    data.ctrl[6] = ctrl[6]
    data.qvel[6:12] = 0.0
    data.qpos[7:13] = arm_target
    policy._carry_cube_with(data)
    mujoco.mj_step(model, data)
    data.qpos[7:13] = arm_target
    policy._carry_cube_with(data)
    return done, info


def _run_policy(model, data, policy, max_steps=350):
    for _ in range(max_steps):
        done, info = _step_once(model, data, policy)
        if done:
            break
    mujoco.mj_forward(model, data)


def test_policy_approaches_cube_at_known_pose():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    cube_xy = np.array([0.6, 0.0])
    policy = ScriptedPickAndPlace(model, data, cube_start_xy=cube_xy)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_center")
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")

    for _ in range(80):
        done, info = _step_once(model, data, policy)
        if done:
            break

    mujoco.mj_forward(model, data)
    ee_xyz = data.site_xpos[site_id]
    cube_xyz = data.xpos[cube_body_id]
    dist = np.linalg.norm(ee_xyz - cube_xyz)
    assert dist < 0.20, f"EE did not approach cube: dist={dist:.3f}"


def test_policy_advances_phases():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array([0.6, 0.0]))
    phases_seen = set()
    for _ in range(250):
        done, info = _step_once(model, data, policy)
        phases_seen.add(info["phase"])
        if done:
            break

    assert 0 in phases_seen, "Never entered APPROACH phase"
    assert len(phases_seen) > 1, f"Policy never advanced past phase 0, saw {phases_seen}"


def test_policy_full_pick_and_place():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array([0.6, 0.0]))
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    target_pos = data.xpos[target_body_id].copy()

    _run_policy(model, data, policy, max_steps=350)

    cube_final = data.xpos[cube_body_id].copy()
    dist_to_target = np.linalg.norm(cube_final[:2] - target_pos[:2])
    assert dist_to_target < 0.12, f"Cube did not reach target: dist={dist_to_target:.3f}, cube={cube_final}, target={target_pos}"
