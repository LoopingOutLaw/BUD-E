"""Tighter confidence sweep at the strongest known setting from env_knob_sweep.

Runs N_INITS seeds × N_ROLLOUTS cube positions, single fixed protocol:
  ensembling=True, ensembling_k=0.4, max_tries=3, max_steps_per_try=80.

Outputs a JSON confidence estimate plus saves the per-seed success array.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "src")

import argparse, json, datetime, time
import numpy as np, mujoco, torch
from bude_vla.envs.so101_mjx import ARM_MODEL_PATH
from bude_vla.env_runner import PolicyRolloutRunner
from bude_vla.models.policy import BUDEPolicy, BUDEConfig


def wilson_ci(s, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = s / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/pick_v4_25k/pick_v4_25k_final.pt")
    p.add_argument("--inits", type=int, default=5)
    p.add_argument("--rollouts", type=int, default=30)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--ensembling-k", type=float, default=0.4)
    p.add_argument("--max-tries", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--ckpt-suffix", default=None)
    args = p.parse_args()

    print(f"loading {args.ckpt}", flush=True)
    t0 = time.time()
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
    print(f"ckpt step={ckpt.get('step')} loss={ckpt['loss_history'][-1][1]:.4f}", flush=True)

    model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
    data = mujoco.MjData(model)
    runner = PolicyRolloutRunner(
        model, img_size=224,
        max_steps_per_try=args.max_steps,
        max_tries=args.max_tries,
        device="cuda", action_lo=lo, action_hi=hi,
        n_history_frames=2,
        ensembling=True, ensembling_k=args.ensembling_k,
    )
    print(f"protocol: ensembling={True} k={args.ensembling_k} "
          f"t={args.max_tries} max_steps={args.max_steps}", flush=True)

    succ_per_seed = []
    trials_per_seed = []
    frames_per_rollout = []
    all_xy = []

    for s in range(args.inits):
        rng_s = np.random.default_rng(args.seed0 + s)
        succ = 0
        tries_used = 0
        seed_xys = []
        for i in range(args.rollouts):
            xy = np.array([
                float(rng_s.uniform(0.55, 0.70)),
                float(rng_s.uniform(-0.10, 0.10)),
            ])
            seed_xys.append(xy)
            mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
            data.qpos[0:3] = [xy[0], xy[1], 0.445]
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(model, data)
            r = runner.run_one(data, policy, xy)
            if r.success:
                succ += 1
            tries_used += r.n_tries
            frames_per_rollout.append(len(r.frames))
            all_xy.append(xy.tolist())
        succ_per_seed.append(succ)
        trials_per_seed.append(tries_used)
        elapsed = time.time() - t0
        eta = elapsed * (args.inits - s - 1) / (s + 1)
        rate = succ / args.rollouts
        print(f"  seed {s}: {succ}/{args.rollouts} ({100*rate:.1f}%) "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    n_total = args.inits * args.rollouts
    s_total = sum(succ_per_seed)
    rate = s_total / n_total
    lo_c, hi_c = wilson_ci(s_total, n_total)
    succ_arr = np.asarray(succ_per_seed, np.float32)
    trials_arr = np.asarray(trials_per_seed, np.int32)
    frames_arr = np.asarray(frames_per_rollout, np.int32)

    print()
    print("=" * 60)
    print(f"OVERALL: {s_total}/{n_total} = {100*rate:.1f}%")
    print(f"wilson 95% CI: [{100*lo_c:.1f}%, {100*hi_c:.1f}%]")
    print(f"per-seed success: {succ_arr.tolist()}")
    print(f"per-seed mean={succ_arr.mean():.1f}/{args.rollouts} std={succ_arr.std():.2f}")
    print(f"trial count mean={trials_arr.mean():.2f}/{args.rollouts*args.max_tries} "
          f"min={int(trials_arr.min())} max={int(trials_arr.max())}")
    print(f"frames mean={frames_arr.mean():.1f} max={int(frames_arr.max())} "
          f"min={int(frames_arr.min())}")
    print("=" * 60)
    runner.close()

    out = {
        "timestamp": datetime.datetime.now().isoformat(),
        "ckpt": args.ckpt,
        "ckpt_step": ckpt.get("step"),
        "ckpt_loss": float(ckpt["loss_history"][-1][1]),
        "protocol": {
            "ensembling": True,
            "ensembling_k": args.ensembling_k,
            "max_tries": args.max_tries,
            "max_steps_per_try": args.max_steps,
        },
        "seeds_inits": args.inits,
        "rollouts_per_init": args.rollouts,
        "total_rollouts": n_total,
        "success_overall": s_total,
        "rate_overall": rate,
        "ci95_low": lo_c,
        "ci95_high": hi_c,
        "per_seed_success": succ_per_seed,
        "per_seed_mean": float(succ_arr.mean()),
        "per_seed_std": float(succ_arr.std()),
        "frames_stats": {
            "mean": float(frames_arr.mean()),
            "min": int(frames_arr.min()),
            "max": int(frames_arr.max()),
        },
        "all_cube_xy": all_xy,
        "is_vision_only": True,
    }
    suffix = args.ckpt_suffix or f"step{ckpt.get('step')}_t{args.max_tries}_k{args.ensembling_k:.2f}"
    os.makedirs("results", exist_ok=True)
    out_path = f"results/confirmation_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
