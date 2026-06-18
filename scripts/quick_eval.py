"""Quick mid-training eval: load latest checkpoint and run 3 rollouts."""
import os, sys, glob
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import torch
sys.path.insert(0, "src")

from bude_vla.envs.so101_mjx import ARM_MODEL_PATH
from bude_vla.env_runner import PolicyRolloutRunner
from bude_vla.models.policy import BUDEPolicy, BUDEConfig

def main():
    # Find latest checkpoint
    ckpts = sorted(glob.glob("checkpoints/pick_v4_25k/*.pt"))
    if not ckpts:
        print("No checkpoints found yet. Training needs more time.")
        return
    latest = ckpts[-1]
    print(f"Loading: {latest}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(latest, map_location=device, weights_only=False)

    cfg = BUDEConfig()
    cfg.img_size = 224
    cfg.patch_size = 16
    cfg.chunk_size = 4
    if "config" in ckpt:
        for k in ["use_dinov2", "use_minilm", "n_history_frames", "chunk_size"]:
            if k in ckpt["config"]:
                setattr(cfg, k, ckpt["config"][k])

    policy = BUDEPolicy(cfg).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    action_lo = np.asarray(ckpt.get("action_norm_lo", None), dtype=np.float32)
    action_hi = np.asarray(ckpt.get("action_norm_hi", None), dtype=np.float32)
    step = ckpt.get("step", "?")
    loss = ckpt.get("loss_history", [None])[-1]
    if loss:
        print(f"  step={step}, loss={loss[1]:.6f}")
    else:
        print(f"  step={step}")

    model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
    data = mujoco.MjData(model)

    runner = PolicyRolloutRunner(
        model, img_size=224, max_steps_per_try=350, max_tries=3,
        device=device, action_lo=action_lo, action_hi=action_hi,
        n_history_frames=cfg.n_history_frames,
    )

    rng = np.random.default_rng(42)
    n_success = 0
    for i in range(3):
        cube_xy = np.array([float(rng.uniform(0.55, 0.70)),
                            float(rng.uniform(-0.10, 0.10))])
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        data.qpos[0:3] = [cube_xy[0], cube_xy[1], 0.445]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        result = runner.run_one(data, policy, cube_xy)
        status = "SUCCESS" if result.success else "FAILED"
        if result.success:
            n_success += 1
        print(f"  rollout {i+1}/3: cube=({cube_xy[0]:.2f},{cube_xy[1]:.2f}) -> {status}")

    print(f"\n=== Quick eval: {n_success}/3 success ({n_success*33}%) ===")
    if n_success >= 1:
        print("The policy is showing signs of life! Let it train overnight.")
    else:
        print("Not succeeding yet, but this is EARLY in training. Loss is still dropping.")
    runner.close()

if __name__ == "__main__":
    main()
