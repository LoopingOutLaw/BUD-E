"""Render a single rollout to MP4 (3x slowed). Cube XY is one we know succeeds."""
import os, sys
sys.path.insert(0, "src")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, torch
import imageio
from bude_vla.envs.so101_mjx import ARM_MODEL_PATH
import bude_vla.env_runner as er
from bude_vla.models.policy import BUDEPolicy, BUDEConfig

ckpt = torch.load("checkpoints/pick_v4_25k/pick_v4_25k_final.pt",
                  map_location="cuda", weights_only=False)
cfg = BUDEConfig()
for k, v in ckpt["config"].items():
    setattr(cfg, k, v)
policy = BUDEPolicy(cfg).to("cuda")
policy.load_state_dict(ckpt["model_state_dict"])
policy.eval()

model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
data = mujoco.MjData(model)
lo = np.asarray(ckpt["action_norm_lo"], np.float32)
hi = np.asarray(ckpt["action_norm_hi"], np.float32)
runner = er.PolicyRolloutRunner(
    model, img_size=224, max_steps_per_try=80, max_tries=3,
    device="cuda", action_lo=lo, action_hi=hi,
    n_history_frames=cfg.n_history_frames,
)

cube_xy = np.array([0.63, -0.09])  # one that succeeded in earlier evals
mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
data.qpos[0:3] = [cube_xy[0], cube_xy[1], 0.445]
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
mujoco.mj_forward(model, data)
r = runner.run_one(data, policy, cube_xy)
print(f"success={r.success} frames={len(r.frames)}")
slow = [f[:, :, :3] for f in r.frames for _ in range(3)]
os.makedirs("demos/videos", exist_ok=True)
imageio.mimsave("demos/videos/my_vla_run.mp4", slow, fps=10, macro_block_size=1)
runner.close()
print("wrote demos/videos/my_vla_run.mp4")
