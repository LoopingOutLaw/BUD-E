#!/usr/bin/env bash
# Live-viewer VLA eval. Usage: bash scripts/run_vla_live.sh
set -e
cd "$(dirname "$0")/.."
unset PYTHONPATH
export MUJOCO_GL=glfw
export PYTHONPATH=src
/home/aditya/.bude-venv/bin/python - <<'PY'
import os, sys
import numpy as np, mujoco, torch
from bude_vla.envs.so101_mjx import ARM_MODEL_PATH
import bude_vla.env_runner as er
from bude_vla.models.policy import BUDEPolicy, BUDEConfig

ckpt = torch.load('checkpoints/pick_v4_25k/pick_v4_25k_final.pt',
                  map_location='cuda', weights_only=False)
cfg = BUDEConfig()
for k, v in ckpt['config'].items():
    setattr(cfg, k, v)
policy = BUDEPolicy(cfg).to('cuda')
policy.load_state_dict(ckpt['model_state_dict'])
policy.eval()

model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
data = mujoco.MjData(model)
lo = np.asarray(ckpt['action_norm_lo'], np.float32)
hi = np.asarray(ckpt['action_norm_hi'], np.float32)
runner = er.PolicyRolloutRunner(
    model, img_size=224, max_steps_per_try=80, max_tries=3,
    device='cuda', action_lo=lo, action_hi=hi,
    n_history_frames=cfg.n_history_frames,
)

cube_xy = np.array([0.66, 0.05])
mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
data.qpos[0:3] = [cube_xy[0], cube_xy[1], 0.445]
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
mujoco.mj_forward(model, data)

try:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as v:
        r = runner.run_one(data, policy, cube_xy, viewer=v)
        print(f'success={r.success}, frames={len(r.frames)}')
except Exception as e:
    print('viewer failed (probably headless):', e)
    r = runner.run_one(data, policy, cube_xy)
    print(f'success={r.success}, frames={len(r.frames)}')
runner.close()
PY
