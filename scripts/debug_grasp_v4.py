"""Debug grasp metrics during GRIP phase — v4 policy."""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mujoco
import numpy as np
from bude_vla.envs.so101_mjx import (
    load_arm_model, default_joint_angles,
    GRIPPER_QPOS_START, CUBE_QPOS_START, CUBE_REST_Z,
)
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace, PHASE_NAMES

def main():
    model = load_arm_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:5] = default_joint_angles(model)
    data.qpos[GRIPPER_QPOS_START] = 1.5
    # Cube properly on pedestal
    data.qpos[CUBE_QPOS_START:CUBE_QPOS_START+3] = [0.30, 0.0, CUBE_REST_Z]
    data.qpos[CUBE_QPOS_START+3:CUBE_QPOS_START+7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array([0.30, 0.0]))

    for step in range(1200):
        ctrl, arm_target, done, info = policy.step(model, data)

        phase = info.get("phase", -1)
        pname = PHASE_NAMES.get(phase, f"phase_{phase}")

        # Print detailed metrics during DESCENT and GRIP phases
        if phase >= 3:  # DESCENT or later
            jaw_qpos = float(data.qpos[GRIPPER_QPOS_START])
            jc = data.site_xpos[policy.jaw_site_id].copy()
            ff = data.site_xpos[policy.ff_site_id].copy()
            cube = data.xpos[policy.cube_body_id].copy()
            midpoint = np.array([(jc[0]+ff[0])/2, (jc[1]+ff[1])/2, min(jc[2], ff[2])])
            dist_to_midpoint = float(np.linalg.norm(cube - midpoint))
            gap = policy.grasp.gap(data)
            has_contact = policy.grasp._has_gripper_cube_contact(model, data)
            lowest_z = policy._lowest_box_z(data)
            attached = info.get("attached", False)

            if step % 20 == 0 or phase != 3:  # Print more often during GRIP
                print(f"step={step:4d} phase={pname:10s} jaw={jaw_qpos:.3f} "
                      f"gap={gap:.4f} dist_mid={dist_to_midpoint:.4f} "
                      f"contact={has_contact} attached={attached} "
                      f"lowest_z={lowest_z:.4f} "
                      f"jc_z={jc[2]:.4f} ff_z={ff[2]:.4f} cube_z={cube[2]:.4f}")

        if done:
            print(f"\nFinal: attached={policy.grasp.state.attached}")
            print(f"Release reason: {policy.grasp.state.release_reason}")
            cube_final = data.xpos[policy.cube_body_id].copy()
            print(f"Cube final pos: {cube_final}")
            break

if __name__ == "__main__":
    main()
