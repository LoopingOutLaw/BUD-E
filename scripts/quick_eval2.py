"""Minimal eval: 10 rollouts, lower memory, faster."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "src")
import numpy as np, mujoco, torch
from bude_vla.envs.so101_mjx import ARM_MODEL_PATH
from bude_vla.env_runner import PolicyRolloutRunner
from bude_vla.models.policy import BUDEPolicy, BUDEConfig

ckpt = torch.load("checkpoints/pick_v4_25k/pick_v4_25k_final.pt",
                   map_location="cuda", weights_only=False)
cfg = BUDEConfig()
for k, v in ckpt.get("config", {}).items():
    setattr(cfg, k, v)

policy = BUDEPolicy(cfg).to("cuda")
policy.load_state_dict(ckpt["model_state_dict"])
policy.eval()
action_lo = np.asarray(ckpt["action_norm_lo"], dtype=np.float32)
action_hi = np.asarray(ckpt["action_norm_hi"], dtype=np.float32)
print(f"loaded step={ckpt.get('step')} loss={ckpt['loss_history'][-1][1]:.4f}")

model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
data = mujoco.MjData(model)
runner = PolicyRolloutRunner(
    model, img_size=224, max_steps_per_try=160, max_tries=5,
    device="cuda", action_lo=action_lo, action_hi=action_hi,
    n_history_frames=cfg.n_history_frames,
    ensembling=True, ensembling_k=0.45,
)

rng = np.random.default_rng(42)
n_succ = 0
for i in range(10):
    cube_xy = np.array([float(rng.uniform(0.55, 0.70)),
                        float(rng.uniform(-0.10, 0.10))])
    mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    data.qpos[0:3] = [cube_xy[0], cube_xy[1], 0.445]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]; mujoco.mj_forward(model, data)
    print(f"--- rollout {i+1}: cube_xy=({cube_xy[0]:.2f},{cube_xy[1]:.2f}) ---")
    r = runner.run_one(data, policy, cube_xy)
    s = "SUCCESS" if r.success else "FAILED"
    if r.success:
        n_succ += 1
    print(f"  rollout {i+1}: cube=({cube_xy[0]:.2f},{cube_xy[1]:.2f}) -> {s} (frames={len(r.frames)})")
print(f"=== {n_succ}/10 ===")
runner.close()
