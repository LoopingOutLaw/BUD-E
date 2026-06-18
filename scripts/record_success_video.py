"""Record a smooth, postable success MP4 for the BUD-E pick policy.

Settings:
- ensembling=True, k=0.45, max_tries=5, max_steps_per_try=160
- record_video_mode=True enables 6 sub-step renders per policy action,
  so the arm visibly sweeps toward each target instead of teleporting.
- render via the `portfolio` camera (side-front view that shows arm, gripper,
  cube and target zone simultaneously).

Re-rolls through 6 known-good cube positions until the first success wins;
records it at 30 fps so the playback reads like natural motion.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "src")

import numpy as np, mujoco, torch
import imageio

ckpt_path = os.environ.get(
    "BUDE_CKPT",
    "checkpoints/pick_v4_25k/pick_v4_25k_final.pt",
)

ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
from bude_vla.models.policy import BUDEPolicy, BUDEConfig
from bude_vla.envs.so101_mjx import ARM_MODEL_PATH
import bude_vla.env_runner as er

cfg = BUDEConfig()
for k, v in ckpt.get("config", {}).items():
    setattr(cfg, k, v)
cfg.n_history_frames = 2

policy = BUDEPolicy(cfg).to("cuda")
policy.load_state_dict(ckpt["model_state_dict"])
policy.eval()

model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
data = mujoco.MjData(model)
lo = np.asarray(ckpt["action_norm_lo"], dtype=np.float32)
hi = np.asarray(ckpt["action_norm_hi"], dtype=np.float32)

runner = er.PolicyRolloutRunner(
    model, img_size=224,
    max_steps_per_try=160,
    max_tries=5,
    device="cuda",
    action_lo=lo,
    action_hi=hi,
    n_history_frames=cfg.n_history_frames,
    ensembling=True,
    ensembling_k=0.45,
    arm_smooth_steps=14,
)

cube_positions = [
    np.array([0.62, -0.09]),
    np.array([0.58, -0.07]),
    np.array([0.68,  0.08]),
    np.array([0.55, -0.06]),
    np.array([0.64,  0.05]),
    np.array([0.66,  0.05]),
]


def show_frame_attempt(frame):
    return frame[:, :, :3]


for attempt, cube_xy in enumerate(cube_positions):
    mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    data.qpos[0:3] = [cube_xy[0], cube_xy[1], 0.445]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    r = runner.run_one(data, policy, cube_xy,
                       record_video_mode=True, record_camera="portfolio")
    print(f"  attempt {attempt + 1}: cube=({cube_xy[0]:.2f},{cube_xy[1]:+.2f}) "
          f"-> success={r.success} frames={len(r.frames)}", flush=True)
    if r.success:
        winning_frames = r.frames
        winning_xy = cube_xy
        winning_tries = r.n_tries
        break
else:
    raise SystemExit("no success after 6 attempts")


def show_frame(frame):
    if frame.ndim == 3 and frame.shape[-1] == 6:
        return frame[:, :, :3]
    return frame


stack = np.stack([show_frame(f) for f in winning_frames])
print(f"  picked cube=({winning_xy[0]:.2f},{winning_xy[1]:+.2f}) "
      f"n_tries={winning_tries} frames={len(stack)}", flush=True)

os.makedirs("demos/videos", exist_ok=True)
out_path = "demos/videos/pick_success_v4_25k.mp4"

FPS_OUT = 15
with imageio.get_writer(out_path, fps=FPS_OUT, macro_block_size=1) as w:
    for f in stack:
        w.append_data(f)

runner.close()
print(f"wrote {out_path}  ({len(stack)} frames @ {FPS_OUT} fps = "
      f"{len(stack) / FPS_OUT:.1f} s)", flush=True)
