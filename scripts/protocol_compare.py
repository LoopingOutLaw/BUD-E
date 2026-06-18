"""Compare eval protocols: legacy vs temporal-ensembling vs EMA.

Reports success rate per protocol at matched seed grid. Targets file outputs
under results/ for later A/B reporting.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "src")

import argparse
import numpy as np, mujoco, torch
from bude_vla.envs.so101_mjx import ARM_MODEL_PATH
from bude_vla.env_runner import PolicyRolloutRunner
from bude_vla.models.policy import BUDEPolicy, BUDEConfig


def run_proto(policy, lo, hi, runner_kwargs, label, xys):
    model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
    data = mujoco.MjData(model)
    runner = PolicyRolloutRunner(model, img_size=224,
                                 max_steps_per_try=80, max_tries=1,
                                 device="cuda", action_lo=lo, action_hi=hi,
                                 n_history_frames=2,
                                 **runner_kwargs)
    succ = 0
    nframes = []
    for xy in xys:
        mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
        data.qpos[0:3] = [xy[0], xy[1], 0.445]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        r = runner.run_one(data, policy, xy)
        if r.success:
            succ += 1
        nframes.append(len(r.frames))
    runner.close()
    print(f"  {label}: {succ}/{len(xys)}  frames_mean={np.mean(nframes):.1f}", flush=True)
    return succ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/pick_v4_25k/pick_v4_25k_final.pt")
    p.add_argument("--inits", type=int, default=3)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--seed0", type=int, default=0)
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

    rng = np.random.default_rng(args.seed0)
    all_xys = []
    for s in range(args.inits):
        rng_s = np.random.default_rng(args.seed0 + s)
        for i in range(args.rollouts):
            all_xys.append(np.array([
                float(rng_s.uniform(0.55, 0.70)),
                float(rng_s.uniform(-0.10, 0.10)),
            ]))

    print(f"protocol comparison on {len(all_xys)} rollouts", flush=True)

    a = run_proto(policy, lo, hi, dict(ensembling=False), "legacy          ", all_xys)
    b = run_proto(policy, lo, hi, dict(ensembling=True, ensembling_k=0.5),
                  "ensembling k=0.5", all_xys)
    c = run_proto(policy, lo, hi, dict(ensembling=True, ensembling_k=0.3),
                  "ensembling k=0.3", all_xys)
    d = run_proto(policy, lo, hi, dict(ensembling=True, ensembling_k=0.7),
                  "ensembling k=0.7", all_xys)

    print()
    print(f"=== summary ({len(all_xys)} rollouts) ===")
    print(f"legacy                : {a}/{len(all_xys)}  {100*a/len(all_xys):.0f}%")
    print(f"ensembling k=0.5      : {b}/{len(all_xys)}  {100*b/len(all_xys):.0f}%")
    print(f"ensembling k=0.3 (new): {c}/{len(all_xys)}  {100*c/len(all_xys):.0f}%")
    print(f"ensembling k=0.7      : {d}/{len(all_xys)}  {100*d/len(all_xys):.0f}%")


if __name__ == "__main__":
    main()
