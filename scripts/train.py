"""BUD-E training script.

Trains the BUDEPolicy on collected reach + push data with flow-matching loss.
Saves checkpoints and produces a single training-progress video.

Usage:
    PYTHONPATH=src python scripts/train.py
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from bude_vla.data.lerobot_v3 import BUDETrainingDataset
from bude_vla.models.policy import BUDEPolicy, BUDEConfig


def collate_fn(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    return out


def build_dataloader(roots: list[str | Path], chunk_size: int = 4,
                     batch_size: int = 32, num_workers: int = 0) -> torch.utils.data.DataLoader:
    all_frames = []
    for root in roots:
        ds = BUDETrainingDataset(root, chunk_size=chunk_size)
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
        pin_memory=True, drop_last=True,
    )
    print(f"  total frames: {len(combined)}, batches/epoch: {len(dl)}")
    return dl


def train(
    data_roots: list[str] | None = None,
    ckpt_dir: str = "/home/aditya/bude_vla/checkpoints",
    video_dir: str = "/home/aditya/bude_vla/demos/videos",
    n_steps: int = 50000,
    batch_size: int = 32,
    chunk_size: int = 4,
    img_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    save_every: int = 5000,
    device: str = "cuda",
    task_name: str = "policy",
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

    policy = BUDEPolicy(cfg).to(device)
    n_params = policy.n_params()
    print(f"Model parameters: {n_params['total']:,}")
    for k, v in n_params.items():
        print(f"  {k}: {v:,}")

    dl = build_dataloader(data_roots, chunk_size=chunk_size,
                          batch_size=batch_size, num_workers=0)

    optimizer = AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.01)

    step = 0
    epoch = 0
    running_loss = 0.0
    t0 = time.time()
    dl_iter = iter(dl)

    loss_history = []

    while step < n_steps:
        try:
            batch = next(dl_iter)
        except StopIteration:
            epoch += 1
            dl_iter = iter(dl)
            batch = next(dl_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        out = policy(batch)
        v_pred = out["velocity"]
        v_target = batch["actions"] - batch["noise"]
        loss = ((v_pred - v_target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
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
            }, ckpt_path)
            print(f"  saved checkpoint: {ckpt_path}")

    final_ckpt = ckpt_dir / f"{task_name}_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,
    }, final_ckpt)
    print(f"Training done in {time.time()-t0:.0f}s. Final checkpoint: {final_ckpt}")

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=64,
                        help="Image resolution for ViT input (default 64, use 224 for hi-res)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--data-root", action="append", default=None,
                        help="Dataset root(s) to train on. May be passed "
                             "multiple times. Defaults to reach_v3 + push_v3.")
    parser.add_argument("--task", default="policy",
                        help="Subdirectory under checkpoints/ for this run, "
                             "and filename prefix (e.g. 'pick', 'reach').")
    args = parser.parse_args()

    roots = args.data_root
    if roots is None:
        roots = ["/home/aditya/bude_vla/data/reach_v3",
                 "/home/aditya/bude_vla/data/push_v3"]

    train(
        data_roots=roots,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        img_size=args.img_size,
        lr=args.lr,
        save_every=args.save_every,
        task_name=args.task,
    )
