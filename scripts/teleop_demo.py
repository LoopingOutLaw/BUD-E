"""Teleoperate the BUD-E arm with keyboard + mouse.

Controls (terminal focus):
      1..6     Select active arm joint (highlighted in heading)
      UP/DN    Move active joint +/-
      ,/.      Slow nudge of active joint
      c        Close gripper
      o        Open gripper
      SPACE    Re-sync target to current pose (handy when stuck)
      r        Toggle RECORDING (saves episode on stop)
      z        Drop last saved episode
      h        Print joint angles + EE pose
      q / Esc  Quit

Mouse: default MuJoCo camera in the viewer window.

Live status (printed in place):
      recording on/off, active joint, joint angles, EE xyz, gripper,
      episode #, frames in current episode

Episodes saved as raw npz + json under <out>/raw/.
Use scripts/fold_teleop.py to fold them into LeRobot v3 layout.

Run:
  XDG_RUNTIME_DIR=/tmp MUJOCO_GL=glfw PYTHONPATH=src \
    /home/aditya/.bude-venv/bin/python /home/aditya/bude_vla/scripts/teleop_demo.py
"""
from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = "/home/aditya/bude_vla/urdf/ur5e_scene.xml"
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "forearm_roll", "wrist_pitch", "wrist_roll",
]
ARM_QPOS_START = 7
GRIP_QPOS_IDX = 13
STEP_DT = 0.005


def get_ee_pos(model, data) -> np.ndarray:
    return data.site("ee_center").xpos.copy()


ANSI = {
    "clr": "\x1b[2J\x1b[H",
    "bold": "\x1b[1m",
    "reset": "\x1b[0m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
}


def print_help():
    sys.stdout.write(ANSI["clr"])
    sys.stdout.write(
        ANSI["bold"] + "BUD-E Teleop" + ANSI["reset"] + "\n"
        + "  1..6       select arm joint (initial: shoulder_pan)\n"
        + "  ↑ ↓        active joint +/-\n"
        + "  , .        slow nudge (±0.02 rad)\n"
        + "  c / o      close / open gripper\n"
        + "  SPACE      re-zero ctrl / re-sync joint targets\n"
        + "  r          toggle RECORDING (saves episode on STOP)\n"
        + "  z          drop last saved episode\n"
        + "  h          print angles + EE pose (newline)\n"
        + "  q / Esc    quit\n"
        + "Mouse inside the viewer window: drag camera, wheel to zoom.\n\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/aditya/bude_vla/data/teleop_v3")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "raw").mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    data.qpos[:] = 0.0
    data.qpos[ARM_QPOS_START:ARM_QPOS_START + 4] = [0.0, -0.5, 0.5, 0.0]
    data.qpos[0:3] = [0.6, 0.0, 0.445]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[GRIP_QPOS_IDX] = 0.0
    mujoco.mj_forward(model, data)

    active_joint = 0
    target_qpos = data.qpos[ARM_QPOS_START:ARM_QPOS_START + 6].copy()
    gripper_target = 0.0
    recording = False
    buf_imgs: list = []
    buf_qpos: list = []
    buf_act: list = []
    episode_idx = 0

    print_help()

    renderer = mujoco.Renderer(model, height=64, width=64)

    def commit_episode():
        nonlocal buf_imgs, buf_qpos, buf_act, episode_idx
        if not buf_imgs:
            return "(empty)"
        ep = {
            "instruction": "reach the target",
            "images": np.stack(buf_imgs),
            "proprio": np.stack(buf_qpos)[:, ARM_QPOS_START:ARM_QPOS_START + 8],
            "actions": np.stack(buf_act),
            "qpos": np.stack(buf_qpos),
        }
        ep_dir = out_root / "raw"
        npz_path = ep_dir / f"episode_{episode_idx:04d}.npz"
        np.savez(
            npz_path,
            instruction=ep["instruction"],
            images=ep["images"],
            qpos=ep["qpos"],
            actions=ep["actions"],
        )
        meta = {
            "episode_index": episode_idx,
            "length": len(buf_imgs),
            "instruction": ep["instruction"],
        }
        (ep_dir / f"episode_{episode_idx:04d}.json").write_text(json.dumps(meta, indent=2))
        saved_idx = episode_idx
        n = len(buf_imgs)
        episode_idx += 1
        buf_imgs, buf_qpos, buf_act = [], [], []
        return f"saved ep {saved_idx} ({n}f)"

    # Put stdin in cbreak mode so we read characters one at a time.
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        last_print = 0.0
        last_step = time.time()
        quit_flag = False
        with mujoco.viewer.launch_passive(model, data) as viewer:
            sys.stdout.write("Teleop session started. Close the window or press q to stop.\n\n")
            sys.stdout.flush()
            while viewer.is_running() and not quit_flag:
                # Drain pending keys.
                while True:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                    if not ready:
                        break
                    ch_raw = os.read(fd, 1).decode("utf-8", errors="ignore")
                    # Arrow keys send ESC [ A/B/C/D as 3-byte sequence.
                    if ch_raw == "\x1b":
                        seq = os.read(fd, 2).decode("utf-8", errors="ignore")
                        ch_raw = {"[A": "UP", "[B": "DOWN",
                                  "[C": "RIGHT", "[D": "LEFT"}.get(seq, "ESC")
                    extra = None
                    if ch_raw in "123456":
                        active_joint = int(ch_raw) - 1
                        extra = f"=> {JOINT_NAMES[active_joint]}"
                    elif ch_raw in ("UP", "DOWN"):
                        sgn = +0.15 if ch_raw == "UP" else -0.15
                        target_qpos[active_joint] += sgn
                        extra = f"{JOINT_NAMES[active_joint]} target={target_qpos[active_joint]:+.2f}"
                    elif ch_raw == ".":
                        target_qpos[active_joint] += 0.02
                        extra = f"nudge +0.02 -> {target_qpos[active_joint]:+.2f}"
                    elif ch_raw == ",":
                        target_qpos[active_joint] -= 0.02
                        extra = f"nudge -0.02 -> {target_qpos[active_joint]:+.2f}"
                    elif ch_raw == "c":
                        gripper_target = -0.04
                        extra = "gripper CLOSE tgt"
                    elif ch_raw == "o":
                        gripper_target = 0.0
                        extra = "gripper OPEN tgt"
                    elif ch_raw == " ":
                        data.ctrl[:] = 0.0
                        target_qpos = data.qpos[ARM_QPOS_START:ARM_QPOS_START + 6].copy()
                        extra = "re-synced targets"
                    elif ch_raw == "r":
                        if recording:
                            saved = commit_episode()
                            extra = f"[rec STOP] {saved}"
                        else:
                            buf_imgs.clear(); buf_qpos.clear(); buf_act.clear()
                            extra = "[rec START]"
                        recording = not recording
                    elif ch_raw == "h":
                        ee = get_ee_pos(model, data)
                        angles = data.qpos[ARM_QPOS_START:ARM_QPOS_START + 6]
                        sys.stdout.write(
                            f"\n  angles={angles}  ee={ee}  "
                            f"grip={data.qpos[GRIP_QPOS_IDX]}\n"
                        )
                        sys.stdout.flush()
                    elif ch_raw in ("q", "Q", "ESC"):
                        if recording:
                            commit_episode()
                        quit_flag = True
                        break
                    if extra:
                        sys.stdout.write("\n  " + extra + "\n")
                        sys.stdout.flush()

                # Step physics.
                now = time.time()
                elapsed = now - last_step
                last_step = now
                err = target_qpos - data.qpos[ARM_QPOS_START:ARM_QPOS_START + 6]
                # PD ctrl to drive joints toward target.
                data.ctrl[:6] = np.clip(err * 60.0, -1.0, 1.0)
                # Drive gripper via the equality constraint (we push the slider).
                gerr = gripper_target - data.qpos[GRIP_QPOS_IDX]
                data.ctrl[6] = float(np.clip(gerr * 40.0, -1.0, 1.0))
                # Also nudge qpos directly to keep teleop snappy under sim stiffness.
                data.qpos[ARM_QPOS_START:ARM_QPOS_START + 6] = (
                    data.qpos[ARM_QPOS_START:ARM_QPOS_START + 6] + np.clip(err * 0.5, -0.05, 0.05)
                )
                mujoco.mj_step(model, data)
                viewer.sync()

                if recording:
                    renderer.update_scene(data)
                    img = renderer.render().copy()
                    buf_imgs.append(img)
                    buf_qpos.append(data.qpos.copy())
                    buf_act.append(data.ctrl.copy())

                # Live status line.
                if now - last_print > 0.15:
                    last_print = now
                    ee = get_ee_pos(model, data)
                    angles = data.qpos[ARM_QPOS_START:ARM_QPOS_START + 6]
                    rec_mark = (ANSI["red"] + "●REC" + ANSI["reset"]) if recording \
                        else (ANSI["green"] + "  RC" + ANSI["reset"])
                    bar = "[" + " ".join(
                        ("*" if i == active_joint else " ") for i in range(6)
                    ) + "]"
                    sys.stdout.write(
                        f"\r  {rec_mark} {bar} "
                        f"q=[{angles[0]:+.2f} {angles[1]:+.2f} {angles[2]:+.2f} "
                        f"{angles[3]:+.2f} {angles[4]:+.2f} {angles[5]:+.2f}] "
                        f"ee=({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:+.2f}) "
                        f"grip={data.qpos[GRIP_QPOS_IDX]:+.3f} "
                        f"ep={episode_idx} fr={len(buf_imgs)}  "
                    )
                    sys.stdout.flush()
                sleep_dt = STEP_DT - (time.time() - now)
                if sleep_dt > 0:
                    time.sleep(sleep_dt)

            if recording:
                commit_episode()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    print(f"\n\nSaved {episode_idx} episodes into {out_root/'raw'}")


if __name__ == "__main__":
    main()

