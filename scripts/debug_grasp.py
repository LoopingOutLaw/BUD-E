"""Quick debug: run one episode and print grasp metrics during GRIP phase."""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mujoco
import numpy as np
from bude_vla.envs.so101_mjx import (
    load_arm_model, default_joint_angles,
    GRIPPER_QPOS_START, CUBE_QPOS_START,
)
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace

def main():
    model = load_arm_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:5] = default_joint_angles(model)
    data.qpos[GRIPPER_QPOS_START] = 1.5
    data.qpos[CUBE_QPOS_START:CUBE_QPOS_START+3] = [0.30, 0.0, 0.010]
    data.qpos[CUBE_QPOS_START+3:CUBE_QPOS_START+7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array([0.30, 0.0]))

    for step in range(900):
        ctrl, arm_target, done, info = policy.step(model, data)
        data.ctrl[:] = ctrl

        for _ in range(3):
            mujoco.mj_step(model, data)

        # Print metrics during GRIP phase
        if info.get("phase") == 3:  # GRIP
            jaw_qpos = float(data.qpos[GRIPPER_QPOS_START])
            gap = policy.grasp.gap(data)
            attached = info.get("attached", False)
            jc = data.site_xpos[policy.jaw_site_id].copy()
            ff = data.site_xpos[policy.ff_site_id].copy() if policy.ff_site_id >= 0 else np.zeros(3)
            cube = data.xpos[policy.cube_body_id].copy()
            midpoint = (jc + ff) / 2.0 if policy.ff_site_id >= 0 else jc
            dist_to_midpoint = float(np.linalg.norm(cube - midpoint))

            # Check contacts
            has_contact = policy.grasp._has_gripper_cube_contact(model, data)

            if step % 20 == 0 or step < 260:
                print(f"step={step} jaw={jaw_qpos:.3f} gap={gap:.4f} "
                      f"dist_mid={dist_to_midpoint:.4f} contact={has_contact} "
                      f"attached={attached} "
                      f"jc_z={jc[2]:.4f} ff_z={ff[2]:.4f} cube_z={cube[2]:.4f}")

        if done:
            break

    print(f"\nFinal: attached={policy.grasp.state.attached}")
    print(f"Release reason: {policy.grasp.state.release_reason}")
    cube_final = data.xpos[policy.cube_body_id].copy()
    print(f"Cube final pos: {cube_final}")

if __name__ == "__main__":
    main()
