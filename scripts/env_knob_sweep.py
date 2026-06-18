"""Sweep 6 single-knob variations of the env on pick_v4_25k_final.pt.

Variants test whether remaining ceiling is from a single hyperparameter still
sub-optimal in the wild.

Knobs:
  A. baseline (ensembling k=0.5)
  B. ensembling k=0.6
  C. ensembling k=0.4
  D. max_tries=3 (3 retries from scratch instead of 1)
  E. ensembling_k=0.5 + max_tries=3 (best of both)
  F. ensembling_k=0.5 + max_tries=2 (cheap retry budget)
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "src")

import argparse, json, datetime
import numpy as np, mujoco, torch
from bude_vla.envs.so101_mjx import ARM_MODEL_PATH
from bude_vla.env_runner import PolicyRolloutRunner
from bude_vla.models.policy import BUDEPolicy, BUDEConfig


def run_variant(policy, lo, hi, xy_grid_per_seed, label, ensembling, k, max_tries):
    model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
    data = mujoco.MjData(model)
    runner = PolicyRolloutRunner(
        model, img_size=224,
        max_steps_per_try=80,
        max_tries=max_tries,
        device="cuda", action_lo=lo, action_hi=hi,
        n_history_frames=2,
        ensembling=ensembling, ensembling_k=k,
    )
    succ = 0
    frames = []
    for s_xys in xy_grid_per_seed:
        for xy in s_xys:
            mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
            data.qpos[0:3] = [xy[0], xy[1], 0.445]
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(model, data)
            r = runner.run_one(data, policy, xy)
            if r.success:
                succ += 1
            frames.append(len(r.frames))
    runner.close()
    rate = succ / sum(len(s) for s in xy_grid_per_seed)
    print(f"  {label:30s}  {succ}/{sum(len(s) for s in xy_grid_per_seed)}  "
          f"{100*rate:4.1f}%  frames_mean={np.mean(frames):.1f}  "
          f"max_tries={max_tries}", flush=True)
    return succ, rate, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/pick_v4_25k/pick_v4_25k_final.pt")
    p.add_argument("--inits", type=int, default=3)
    p.add_argument("--rollouts", type=int, default=10)
    p.add_argument("--seed0", type=int, default=42)
    args = p.parse_args()

    print(f"loading {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    cfg = BUDEConfig()
    for k, v in ckpt.get("config", {}).items():
        setattr(cfg, k, v)
    cfg.n_history_frames = 2
    policy = BUDEPolicy(cfg).to("cuda")
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    lo = np.asarray(ckpt["action_norm_lo"], dtype=np.float32)
    hi = np.asarray(ckpt["action_norm_hi"], dtype=np.float32)

    grid = []
    for s in range(args.inits):
        rng_s = np.random.default_rng(args.seed0 + s)
        seed_xys = []
        for i in range(args.rollouts):
            seed_xys.append(np.array([
                float(rng_s.uniform(0.55, 0.70)),
                float(rng_s.uniform(-0.10, 0.10)),
            ]))
        grid.append(seed_xys)

    total = args.inits * args.rollouts
    print(f"sweeping {len(grid)} seeds × {args.rollouts} XYs = {total} per variant", flush=True)

    results = {}
    for label, ensembling, k, tries in [
        ("A. ensembling k=0.5 t=1",      True,  0.5, 1),
        ("B. ensembling k=0.6 t=1",      True,  0.6, 1),
        ("C. ensembling k=0.4 t=1",      True,  0.4, 1),
        ("E. ensembling k=0.5 t=3",      True,  0.5, 3),
        ("F. ensembling k=0.4 t=3",      True,  0.4, 3),
        ("G. ensembling k=0.6 t=2",      True,  0.6, 2),
    ]:
        s, r, fr = run_variant(policy, lo, hi, grid, label, ensembling, k, tries)
        results[label] = {"successes": s, "rate": r,
                          "frames_mean": float(np.mean(fr)),
                          "k": k, "tries": tries}

    out = {
        "timestamp": datetime.datetime.now().isoformat(),
        "ckpt": args.ckpt,
        "seeds_inits": args.inits,
        "seeds_rollouts": args.rollouts,
        "total_rollouts_per_variant": total,
        "variants": results,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/env_knob_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/env_knob_sweep.json")


if __name__ == "__main__":
    main()
