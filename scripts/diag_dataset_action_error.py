"""Measure checkpoint action error on recorded dataset frames.

This diagnostic answers one narrow question: can the policy reproduce the
recorded action chunks when fed the exact dataset observations used for
training? If this fails badly, closed-loop eval cannot work regardless of
physics details.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from bude_vla.data.action_normalization import denormalize_actions
from bude_vla.data.lerobot_v3 import BUDETrainingDataset
from bude_vla.models.policy import BUDEConfig, BUDEPolicy


def load_policy(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("config", {})
    cfg = BUDEConfig()
    cfg.use_dinov2 = saved_cfg.get("use_dinov2", False)
    cfg.use_minilm = saved_cfg.get("use_minilm", False)
    cfg.dinov2_finetune_blocks = saved_cfg.get("dinov2_finetune_blocks", 4)
    cfg.n_history_frames = saved_cfg.get("n_history_frames", 1)
    cfg.chunk_size = saved_cfg.get("chunk_size", 4)
    cfg.img_size = saved_cfg.get("img_size", 224)
    cfg.action_dim = saved_cfg.get("action_dim", 6)
    cfg.state_dim = saved_cfg.get("state_dim", 6)
    cfg.use_bc_head = saved_cfg.get("use_bc_head", False)
    cfg.use_visual_action_cond = saved_cfg.get("use_visual_action_cond", False)
    cfg.use_context_action_head = saved_cfg.get("use_context_action_head", False)
    cfg.use_perception = saved_cfg.get("use_perception", False)
    cfg.use_perception_action_cond = saved_cfg.get("use_perception_action_cond", False)
    cfg.perception_dim = saved_cfg.get("perception_dim", 3)

    policy = BUDEPolicy(cfg).to(device)
    policy.load_state_dict(ckpt.get("ema_state_dict") or ckpt["model_state_dict"])
    policy.eval()

    lo = np.asarray(ckpt["action_norm_lo"], dtype=np.float32)
    hi = np.asarray(ckpt["action_norm_hi"], dtype=np.float32)
    return policy, cfg, lo, hi, ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pick_v12_dinov2/pick_v12_dinov2_step_250000.pt")
    ap.add_argument("--data-root", default="data/pick_v12")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--samples-per-episode", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, cfg, lo, hi, ckpt = load_policy(args.ckpt, device)
    print(f"checkpoint step={ckpt.get('step')} img={cfg.img_size} chunk={cfg.chunk_size} "
          f"history={cfg.n_history_frames} state={cfg.state_dim} action={cfg.action_dim}")

    ds = BUDETrainingDataset(
        Path(args.data_root),
        chunk_size=cfg.chunk_size,
        n_history_frames=cfg.n_history_frames,
        lazy_videos=True,
        normalize=True,
    ).read()
    rng = np.random.default_rng(args.seed)

    rows = []
    max_ep = min(args.episodes, len(ds._episodes))
    with torch.no_grad():
        for ep_i in range(max_ep):
            ep = ds._episodes[ep_i]
            length = ep["length"]
            if length <= 1:
                continue
            frame_ids = np.linspace(0, length - 1, args.samples_per_episode, dtype=int)
            # Add a little randomness but keep early/mid/late coverage.
            jitter = rng.integers(-3, 4, size=frame_ids.shape)
            frame_ids = np.clip(frame_ids + jitter, 0, length - 1)
            for frame_in_ep in frame_ids:
                idx = ds._cum_frames[ep_i] + int(frame_in_ep)
                item = ds[idx]
                batch = {
                    "images": item["images"].unsqueeze(0).to(device),
                    "text_ids": item["text_ids"].unsqueeze(0).to(device),
                    "instruction": [item["instruction"]],
                    "proprio": item["proprio"].unsqueeze(0).to(device),
                    "perception": item["perception"].unsqueeze(0).to(device),
                    "domain_id": item["domain_id"].view(1).to(device),
                }
                pred_norm = policy.sample(batch)[0].detach().cpu().numpy()
                pred = denormalize_actions(pred_norm, lo, hi)
                gt_norm = item["actions"].detach().cpu().numpy()
                gt = denormalize_actions(gt_norm, lo, hi)
                mask = item["mask"].detach().cpu().numpy().astype(bool)
                err = np.abs(pred[mask] - gt[mask])
                rows.append((ep_i, int(frame_in_ep), float(err.mean()),
                             float(err[:, :5].mean()), float(err[:, 5].mean()),
                             pred[0], gt[0]))

    if not rows:
        raise RuntimeError("no rows sampled")

    mean_all = np.mean([r[2] for r in rows])
    mean_arm = np.mean([r[3] for r in rows])
    mean_grip = np.mean([r[4] for r in rows])
    print(f"sampled={len(rows)} mean_abs_error all={mean_all:.4f} arm={mean_arm:.4f} grip={mean_grip:.4f}")
    print("\nfirst-action comparisons:")
    for ep_i, frame, mae, arm_mae, grip_mae, pred0, gt0 in rows[:20]:
        print(f"  ep={ep_i:03d} frame={frame:04d} mae={mae:.4f} "
              f"arm={arm_mae:.4f} grip={grip_mae:.4f}")
        print(f"    pred0 arm={np.array2string(pred0[:5], precision=3)} grip={pred0[5]:+.3f}")
        print(f"    gt0   arm={np.array2string(gt0[:5], precision=3)} grip={gt0[5]:+.3f}")


if __name__ == "__main__":
    main()
