"""Live-viewer smoke test for BUD-E.

Run with the X11 display env vars set so the glfw viewer can find the screen:
    MUJOCO_GL=glfw XDG_RUNTIME_DIR=/tmp \
      PYTHONPATH=src /home/aditya/.bude-venv/bin/python scripts/check_viewer.py [--seconds 8]

A window should pop up showing the UR5e scene jiggling for the requested seconds,
then auto-close.
"""
from __future__ import annotations

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "/home/aditya/bude_vla/urdf/ur5e_scene.xml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=8.0)
    args = p.parse_args()

    if os.environ.get("MUJOCO_GL", "egl") != "glfw":
        print("Hint: prepend MUJOCO_GL=glfw — running anyway, may not show window.")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    print(f"Loaded {MODEL_PATH}, nq={model.nq}, nu={model.nu}, dt={model.opt.timestep:.3f}s")

    n_done = 0
    wall_start = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer opened — physics stepping now.")
        next_print = wall_start + 1.0
        while viewer.is_running() and (time.time() - wall_start) < args.seconds:
            step_start = time.time()
            data.ctrl[:] = np.random.uniform(-0.2, 0.2, size=model.nu)
            mujoco.mj_step(model, data)
            with viewer.lock():
                pass
            viewer.sync()
            n_done += 1
            if time.time() > next_print:
                fps = n_done / (time.time() - wall_start)
                print(f"  step {n_done:5d} | sim_time {data.time:5.2f}s | "
                      f"realtime {fps:5.1f} steps/s")
                next_print = time.time() + 1.0
            elapsed_step = time.time() - step_start
            dt = model.opt.timestep - elapsed_step
            if dt > 0:
                time.sleep(dt)

    total = time.time() - wall_start
    print(f"\n[{n_done} steps in {total:.01f}s -> {n_done/total:.1f} steps/s]")


if __name__ == "__main__":
    main()
