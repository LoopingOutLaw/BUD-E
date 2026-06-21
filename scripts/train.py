"""BUD-E training script.

Trains the BUDEPolicy on collected reach + push data with flow-matching loss.
Saves checkpoints and produces a single training-progress video.

Usage:
    PYTHONPATH=src python scripts/train.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

from bude_vla.data.lerobot_v3 import BUDETrainingDataset
from bude_vla.models.policy import BUDEPolicy, BUDEConfig


def collate_fn(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    out = {}
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, str):
            # Leave language instruction as list[str] — never tensor-stack strings.
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def _detect_dim(roots, key):
    """Read dataset shape for `key` from the first valid info.json."""
    for root in roots:
        info_path = Path(root) / "meta" / "info.json"
        if info_path.exists():
            try:
                meta = json.loads(info_path.read_text())
                feat = meta.get("features", {}).get(key, {})
                shape = feat.get("shape", [])
                if isinstance(shape, list) and shape:
                    return int(shape[0])
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return None


def _detect_action_dim(roots: list) -> int | None:
    return _detect_dim(roots, "action")


def _detect_state_dim(roots: list) -> int | None:
    return _detect_dim(roots, "observation.state")


def build_dataloader(roots: list[str | Path], chunk_size: int = 4,
                     batch_size: int = 32, num_workers: int = 0,
                     augment: bool = False,
                     n_history_frames: int = 1) -> torch.utils.data.DataLoader:
    all_frames = []
    for root in roots:
        ds = BUDETrainingDataset(root, chunk_size=chunk_size, augment=augment,
                                 n_history_frames=n_history_frames)
        ds.read()
        all_frames.append(ds)
        print(f"  loaded {len(ds)} frames from {root}")

    if len(all_frames) == 1:
        combined = all_frames[0]
    else:
        combined = torch.utils.data.ConcatDataset(all_frames)

    dl = torch.utils.data.DataLoader(
        combined, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=False, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    print(f"  total frames: {len(combined)}, batches/epoch: {len(dl)}, "
          f"history_frames: {n_history_frames}")
    return dl


def train(
    data_roots: list[str] | None = None,
    ckpt_dir: str = "/home/aditya/bude_vla/checkpoints",
    video_dir: str = "/home/aditya/bude_vla/demos/videos",
    n_steps: int = 50000,
    batch_size: int = 32,
    grad_accum_steps: int = 1,
    chunk_size: int = 4,
    img_size: int = 64,
    lr: float = 3e-4,
    backbone_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    save_every: int = 5000,
    device: str = "cuda",
    augment: bool = False,
    task_name: str = "policy",
    resume: str | None = None,
    num_workers: int = 0,
    use_dinov2: bool = False,
    use_minilm: bool = False,
    n_history_frames: int = 1,
):
    if data_roots is None:
        data_roots = ["/home/aditya/bude_vla/data/reach_v3",
                      "/home/aditya/bude_vla/data/push_v3"]
    ckpt_dir = Path(ckpt_dir) / task_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    cfg = BUDEConfig()
    cfg.img_size = img_size
    cfg.patch_size = 16
    cfg.chunk_size = chunk_size
    cfg.use_dinov2 = use_dinov2
    cfg.use_minilm = use_minilm
    cfg.n_history_frames = n_history_frames
    # Auto-detect action/state dims from dataset if possible; fall back to 6 (SO-101 5-arm + 1-grip).
    cfg.action_dim = action_dim_override if (action_dim_override := _detect_action_dim(data_roots)) else 6
    cfg.state_dim  = state_dim_override  if (state_dim_override  := _detect_state_dim(data_roots))  else 6

    policy = BUDEPolicy(cfg).to(device)
    n_params = policy.n_params()
    print(f"Model parameters: {n_params['total']:,}")
    for k, v in n_params.items():
        print(f"  {k}: {v:,}")

    dl = build_dataloader(data_roots, chunk_size=chunk_size,
                          batch_size=batch_size, num_workers=num_workers,
                          augment=augment,
                          n_history_frames=n_history_frames)

    from bude_vla.data.action_normalization import load_action_stats, DEFAULT_LO, DEFAULT_HI
    _action_lo, _action_hi = DEFAULT_LO.copy(), DEFAULT_HI.copy()
    for _root in data_roots:
        _info = Path(_root) / "meta" / "info.json"
        _lo, _hi = load_action_stats(_info)
        if not (np.array_equal(_lo, DEFAULT_LO) and np.array_equal(_hi, DEFAULT_HI)):
            _action_lo, _action_hi = _lo, _hi
            break
    print(f"  action_norm lo={_action_lo[:3]}  hi={_action_hi[:3]}")

    # Differential LR: pretrained DINOv2 backbone gets backbone_lr,
    # new modules (proj, text, proprio, backbone transformer, action head) get lr.
    pretrained_params = []
    new_params = []
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        if use_dinov2 and "vision.backbone" in name:
            pretrained_params.append(p)
        else:
            new_params.append(p)
    param_groups = [
        {"params": pretrained_params, "lr": backbone_lr},
        {"params": new_params, "lr": lr},
    ]
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    print(f"  pretrained backbone params: {sum(p.numel() for p in pretrained_params):,} (lr={backbone_lr})")
    print(f"  new module params: {sum(p.numel() for p in new_params):,} (lr={lr})")
    print(f"  effective batch size: {batch_size * grad_accum_steps}")

    scaler = GradScaler()

    # Scheduler in optimizer-update units (not microbatch units).
    # With grad_accum_steps > 1, scheduler.step() fires once per accumulation
    # cycle, so the schedule horizon must match optimizer updates, not microbatches.
    total_opt_steps = max(1, n_steps // grad_accum_steps)
    warmup_opt_steps = max(1, 2000 // grad_accum_steps)
    def lr_lambda(opt_step):
        if opt_step < warmup_opt_steps:
            return opt_step / warmup_opt_steps
        progress = (opt_step - warmup_opt_steps) / max(1, total_opt_steps - warmup_opt_steps)
        progress = min(progress, 1.0)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    step = 0
    epoch = 0
    running_loss = 0.0
    t0 = time.time()
    dl_iter = iter(dl)
    loss_history = []

    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        loss_history = ckpt.get("loss_history", [])
        # Fast-forward scheduler using optimizer-update units (not microbatch)
        opt_steps_done = step // grad_accum_steps
        for _ in range(opt_steps_done):
            scheduler.step()
        running_loss = 0.0
        dl_iter = iter(dl)
        print(f"  resumed from step {step}, lr={scheduler.get_last_lr()[0]:.2e}, "
              f"loss_hist entries={len(loss_history)}")


    while step < n_steps:
        if step % grad_accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        try:
            batch = next(dl_iter)
        except StopIteration:
            epoch += 1
            dl_iter = iter(dl)
            batch = next(dl_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with autocast():
            out = policy(batch)
            v_pred = out["velocity"]
            v_target = batch["actions"] - batch["noise"]
            mask = batch["mask"].unsqueeze(-1)  # (B, chunk_size, 1)
            loss = (((v_pred - v_target) ** 2) * mask).sum() / (mask.sum() * v_pred.shape[-1])
            loss_scaled = loss / grad_accum_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        running_loss += loss.item()
        step += 1

        if step % 100 == 0:
            avg = running_loss / 100
            elapsed = time.time() - t0
            sps = step / elapsed
            loss_history.append((step, avg))
            print(f"step {step:6d} | loss {avg:.6f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                  f"{sps:.1f} steps/s | epoch {epoch}")
            running_loss = 0.0

        if step % save_every == 0:
            ckpt_path = ckpt_dir / f"{task_name}_step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_history": loss_history,
                "action_norm_lo": _action_lo.tolist(),
                "action_norm_hi": _action_hi.tolist(),
                "config": {
                    "use_dinov2": cfg.use_dinov2,
                    "use_minilm": cfg.use_minilm,
                    "n_history_frames": cfg.n_history_frames,
                    "img_size": cfg.img_size,
                    "chunk_size": cfg.chunk_size,
                    "action_dim": cfg.action_dim,
                    "state_dim": cfg.state_dim,
                },
            }, ckpt_path)
            print(f"  saved checkpoint: {ckpt_path}")

    # Flush any remaining accumulated gradients
    if step % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    final_ckpt = ckpt_dir / f"{task_name}_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,
        "action_norm_lo": _action_lo.tolist(),
        "action_norm_hi": _action_hi.tolist(),
        "config": {
            "use_dinov2": cfg.use_dinov2,
            "use_minilm": cfg.use_minilm,
            "n_history_frames": cfg.n_history_frames,
            "img_size": cfg.img_size,
            "chunk_size": cfg.chunk_size,
            "action_dim": cfg.action_dim,
            "state_dim": cfg.state_dim,
        },
    }, final_ckpt)
    print(f"Training done in {time.time()-t0:.0f}s. Final checkpoint: {final_ckpt}")

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps. Effective batch = "
                             "batch_size * grad_accum_steps. Use 4 with "
                             "--batch-size 8 to simulate batch 32 on 8GB GPU.")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=64,
                        help="Image resolution for ViT input (default 64, use 224 for hi-res)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for new modules (backbone, action head, etc.)")
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="Learning rate for pretrained DINOv2 backbone. "
                             "Lower than --lr to preserve pretrained features.")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--data-root", action="append", default=None,
                        help="Dataset root(s) to train on. May be passed "
                             "multiple times. Defaults to reach_v3 + push_v3.")
    parser.add_argument("--task", default="policy",
                        help="Subdirectory under checkpoints/ for this run, "
                             "and filename prefix (e.g. 'pick', 'reach').")
    parser.add_argument("--augment", action="store_true",
                        help="Enable image augmentation (random crop + "
                             "brightness jitter) on the training dataset.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0=single-thread). Use 2-4 "
                             "to keep GPU fed when augmentation is on.")
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint to resume training from. "
                             "Restores model, optimizer, step counter, and "
                             "loss history; continues until --n-steps.")
    parser.add_argument("--use-dinov2", action="store_true",
                        help="Replace from-scratch ViT with frozen pretrained "
                             "DINOv2-small backbone (P0 architecture review fix).")
    parser.add_argument("--use-minilm", action="store_true",
                        help="Replace TinyTextEncoder with frozen pretrained "
                             "MiniLM (sentence-transformers/all-MiniLM-L6-v2).")
    parser.add_argument("--n-history-frames", type=int, default=1,
                        help="Number of stacked history frames per observation. "
                             "1=Markovian (default), 2=allows velocity inference.")
    args = parser.parse_args()

    roots = args.data_root
    if roots is None:
        roots = ["/home/aditya/bude_vla/data/reach_v3",
                 "/home/aditya/bude_vla/data/push_v3"]

    train(
        data_roots=roots,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        chunk_size=args.chunk_size,
        img_size=args.img_size,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        save_every=args.save_every,
        augment=args.augment,
        resume=args.resume,
        task_name=args.task,
        num_workers=args.num_workers,
        use_dinov2=args.use_dinov2,
        use_minilm=args.use_minilm,
        n_history_frames=args.n_history_frames,
    )
