"""Multi-seed variance eval for BUD-E policy rollouts.

Runs N_INITS re-seeds × N_ROLLOUTS_PER_INIT rollouts from random cube positions,
records success rate + per-rollout frame count summary.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "src")

import argparse
import numpy as np, mujoco, torch
from bude_vla.envs.so101_mjx import ARM_MODEL_PATH
from bude_vla.env_runner import PolicyRolloutRunner
from bude_vla.models.policy import BUDEPolicy, BUDEConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/pick_v4_25k/pick_v4_25k_final.pt")
    p.add_argument("--inits", type=int, default=5, help="number of seeded runs")
    p.add_argument("--rollouts", type=int, default=10, help="rollouts per seed")
    p.add_argument("--max-tries", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--x-min", type=float, default=0.55)
    p.add_argument("--x-max", type=float, default=0.70)
    p.add_argument("--y-min", type=float, default=-0.10)
    p.add_argument("--y-max", type=float, default=0.10)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--ckpt-suffix", default=None)
    args = p.parse_args()

    print(f"loading ckpt={args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    cfg = BUDEConfig()
    for k, v in ckpt.get("config", {}).items():
        setattr(cfg, k, v)
    print(f"ckpt step={ckpt.get('step')} loss={ckpt['loss_history'][-1][1]:.4f}", flush=True)

    policy = BUDEPolicy(cfg).to("cuda")
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    lo = np.asarray(ckpt["action_norm_lo"], dtype=np.float32)
    hi = np.asarray(ckpt["action_norm_hi"], dtype=np.float32)

    model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
    data = mujoco.MjData(model)
    runner = PolicyRolloutRunner(
        model, img_size=224,
        max_steps_per_try=args.max_steps,
        max_tries=args.max_tries,
        device="cuda", action_lo=lo, action_hi=hi,
        n_history_frames=cfg.n_history_frames,
    )

    successes_per_seed = []
    frames_per_rollout = []
    cube_xys = []
    results = []

    for s in range(args.inits):
        rng = np.random.default_rng(args.init_seed + s)
        succ = 0
        seed_xys = []
        for i in range(args.rollouts):
            xy = np.array([
                float(rng.uniform(args.x_min, args.x_max)),
                float(rng.uniform(args.y_min, args.y_max)),
            ])
            seed_xys.append(xy)
            mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
            data.qpos[0:3] = [xy[0], xy[1], 0.445]
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(model, data)
            r = runner.run_one(data, policy, xy)
            ok = bool(r.success)
            nframes = len(r.frames)
            if ok:
                succ += 1
            frames_per_rollout.append(nframes)
            results.append((s, i, xy[0], xy[1], ok, nframes))
            print(f"  seed {s:02d} r{i:02d} xy=({xy[0]:.2f},{xy[1]:+.2f}) -> "
                  f"{'OK ' if ok else 'X  '} n={nframes}", flush=True)
        successes_per_seed.append(succ)
        print(f"--- seed {s}: {succ}/{args.rollouts} ---", flush=True)
        cube_xys.append(seed_xys)

    runner.close()

    total = args.inits * args.rollouts
    overall = sum(successes_per_seed)
    succ_arr = np.asarray(successes_per_seed, np.float32)
    frames_arr = np.asarray(frames_per_rollout, np.float32)
    print()
    print(f"=== {overall}/{total}  ({100.0*overall/total:.1f}%) ===")
    print(f"per-seed {succ_arr.tolist()}  mean={succ_arr.mean():.2f} std={succ_arr.std():.2f}")
    print(f"frames: min={frames_arr.min():.0f} max={frames_arr.max():.0f} "
          f"mean={frames_arr.mean():.1f} p90={np.percentile(frames_arr,90):.0f}")
    suffix = args.ckpt_suffix or str(ckpt.get("step"))
    out_npz = f"results/multi_seed_{suffix}.npz"
    np.savez(
        out_npz,
        per_seed_success=succ_arr,
        frame_counts=frames_arr,
        cube_xys=np.array([r[2:4] for r in results], np.float32),
        results=np.array([(r[0], r[1], r[4], r[5]) for r in results], np.float32),
        ckpt=args.ckpt,
        cfg=ckpt.get("config", {}),
    )
    print(f"saved {out_npz}")


if __name__ == "__main__":
    main()
