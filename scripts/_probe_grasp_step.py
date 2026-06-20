#!/usr/bin/env python
"""Probe per-step grasp state during a single scripted episode.

Writes a CSV-ish dump of (step, phase, jaw_qpos, gap, contact) to stdout so we
can diagnose why attach isn't firing. Useful when a single attempt is enough —
21 episodes' worth of summary isn't going to tell us which gate is failing.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse

import mujoco
import numpy as np

from bude_vla.envs.so101_mjx import (
    load_arm_model, default_joint_angles, CUBE_QPOS_START,
)
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = load_arm_model()
    rng = np.random.default_rng(args.seed)
    cx = float(rng.uniform(0.285, 0.315))
    cy = float(rng.uniform(-0.015, 0.015))

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:5] = default_joint_angles(model)
    data.qpos[5] = 1.5
    data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [cx, cy, 0.010]  # cube half-extent on world floor
    data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array([cx, cy]))

    print(f"{'step':>4} {'phase':>5} {'pstp':>4} {'jaw_q':>6} {'gap':>6} {'contact':>7} {'xc':>7} {'yc':>7} {'zc':>6}")
    for step in range(600):
        ctrl, _, done, info = policy.step(model, data)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        jaw_q = float(data.qpos[5])
        gap = policy.grasp.gap(data)
        contact = policy.grasp._has_jaw_ball_contact(model, data)
        cube_xyz = data.xpos[policy.cube_body_id].copy()

        if 50 <= step <= 360 or info.get("phase") not in (0,):
            print(f"{step:>4} {info['phase']:>5} {info['phase_step']:>4} "
                  f"{jaw_q:>6.3f} {gap*1000:>5.1f}m {int(contact):>7} "
                  f"{cube_xyz[0]:>7.3f} {cube_xyz[1]:>7.3f} {cube_xyz[2]:>6.3f}")

        if done:
            break

    print(f"final: attached={policy.attached} reason={policy.grasp.state.release_reason}")


if __name__ == "__main__":
    main()
