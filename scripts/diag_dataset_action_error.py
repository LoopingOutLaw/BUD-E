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
    cfg.use_direct_proprio_action_cond = saved_cfg.get("use_direct_proprio_action_cond", False)
    cfg.perception_dim = saved_cfg.get("perception_dim", 3)
    cfg.input_feature_norm = saved_cfg.get("input_feature_norm", "layernorm")
    cfg.use_gripper_trigger_head = saved_cfg.get("use_gripper_trigger_head", False)
    cfg.gripper_trigger_threshold = saved_cfg.get("gripper_trigger_threshold", 0.5) or 0.5
    cfg.gripper_trigger_close_value = saved_cfg.get("gripper_trigger_close_value", -1.0) or -1.0
    cfg.action_space = saved_cfg.get("action_space", "joint_abs")
    cfg.ee_delta_scale = saved_cfg.get("ee_delta_scale", 0.05) or 0.05

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
    ap.add_argument("--frame-cache", default=None,
                    help="Bounded frame-cache directory used by the training run.")
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
        frame_cache=args.frame_cache,
    ).read()
    rng = np.random.default_rng(args.seed)

    sample_points: list[tuple[int, int, int]] = []
    if ds._cache_global_indices is not None:
        n_samples = min(
            len(ds), max(1, args.episodes * args.samples_per_episode)
        )
        cache_rows = np.sort(rng.choice(len(ds), size=n_samples, replace=False))
        cum_frames = np.asarray(ds._cum_frames, dtype=np.int64)
        for cache_row in cache_rows:
            global_idx = int(ds._cache_global_indices[int(cache_row)])
            ep_i = int(np.searchsorted(cum_frames, global_idx, side="right") - 1)
            frame_in_ep = global_idx - int(cum_frames[ep_i])
            sample_points.append((ep_i, frame_in_ep, int(cache_row)))
    else:
        max_ep = min(args.episodes, len(ds._episodes))
        for ep_i in range(max_ep):
            ep = ds._episodes[ep_i]
            length = ep["length"]
            if length <= 1:
                continue
            frame_ids = np.linspace(
                0, length - 1, args.samples_per_episode, dtype=int
            )
            jitter = rng.integers(-3, 4, size=frame_ids.shape)
            frame_ids = np.clip(frame_ids + jitter, 0, length - 1)
            sample_points.extend(
                (ep_i, int(frame), ds._cum_frames[ep_i] + int(frame))
                for frame in frame_ids
            )

    rows = []
    with torch.no_grad():
        for ep_i, frame_in_ep, item_idx in sample_points:
            item = ds[item_idx]
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
            rows.append((ep_i, frame_in_ep, float(err.mean()),
                         float(err[:, :-1].mean()), float(err[:, -1].mean()),
                         err.mean(axis=0), pred[0], gt[0]))
    if not rows:
        raise RuntimeError("no rows sampled")

    mean_all = np.mean([r[2] for r in rows])
    mean_motion = np.mean([r[3] for r in rows])
    mean_grip = np.mean([r[4] for r in rows])
    per_dim = np.mean(np.stack([r[5] for r in rows]), axis=0)
    print(f"sampled={len(rows)} mean_abs_error all={mean_all:.4f} "
          f"motion={mean_motion:.4f} grip={mean_grip:.4f}")
    print(f"per_dim_mae={np.array2string(per_dim, precision=5)} "
          f"action_space={cfg.action_space}")
    print("\nfirst-action comparisons:")
    for ep_i, frame, mae, motion_mae, grip_mae, _per_dim, pred0, gt0 in rows[:20]:
        print(f"  ep={ep_i:03d} frame={frame:04d} mae={mae:.4f} "
              f"motion={motion_mae:.4f} grip={grip_mae:.4f}")
        print(f"    pred0 motion={np.array2string(pred0[:-1], precision=4)} "
              f"grip={pred0[-1]:+.3f}")
        print(f"    gt0   motion={np.array2string(gt0[:-1], precision=4)} "
              f"grip={gt0[-1]:+.3f}")


if __name__ == "__main__":
    main()
