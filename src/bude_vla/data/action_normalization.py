"""Per-dim action normalization/denormalization for VLA training.

Recorded actions are heterogeneous:
- dims 0..4: arm joint targets in radians
  (IK solver internally clips to ±π but training data spans the full reachable
  range, so observed extremes set the scale).
- dim 5: gripper actuator target in the MJCF actuator control range.

We compute dataset-wide per-dim min/max at write time (one pass over all
episodes) and stash them in `meta/info.json` under `action_normalization`.
Train and eval should call `normalize_actions()` / `denormalize_actions()`
with the same `meta` block so recorded/predicted actions live in the same
mathematical space.

Convention: `a_norm = 2 * (a - lo) / (hi - lo) - 1`, mapping [lo, hi] → [-1, 1].
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


# Default lo/hi used when no empirical stats are available.
# SO-101: 5 arm joints + 1 gripper = 6D action space.
DEFAULT_LO = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -1.5],
                      dtype=np.float32)
DEFAULT_HI = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, 1.5],
                      dtype=np.float32)


def compute_action_stats(actions_iter: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Iterate over per-episode action arrays (T, action_dim) and return dim-wise (lo, hi).

    Uses the empirical min/max across all frames. For real gripper values this
    collapses to {-1, 0, 1} if the scripted policy always emits integer commands.
    """
    global_lo: np.ndarray | None = None
    global_hi: np.ndarray | None = None
    for arr in actions_iter:
        arr = np.asarray(arr, dtype=np.float32)
        if global_lo is None:
            global_lo = arr.min(axis=0)
            global_hi = arr.max(axis=0)
        else:
            global_lo = np.minimum(global_lo, arr.min(axis=0))
            global_hi = np.maximum(global_hi, arr.max(axis=0))
    if global_lo is None:
        return DEFAULT_LO.copy(), DEFAULT_HI.copy()
    return global_lo.astype(np.float32), global_hi.astype(np.float32)


def pad_scale(lo: float, hi: float, margin: float = 0.02) -> tuple[float, float]:
    """Slightly widen [lo, hi] to leave headroom for clipping during training."""
    span = hi - lo if hi > lo else 1e-6
    pad = span * margin
    return float(lo - pad), float(hi + pad)


def normalize_actions(actions: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Map actions from natural [lo, hi] to [-1, 1] per dim.
    `actions` shape (.., D). `lo`/`hi` shape (D,).
    """
    diff = np.where(hi > lo, hi - lo, 1.0).astype(np.float32)
    return 2.0 * (actions.astype(np.float32) - lo.astype(np.float32)) / diff - 1.0


def denormalize_actions(actions: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Inverse of `normalize_actions`. Map [-1, 1] → [lo, hi] per dim."""
    diff = np.where(hi > lo, hi - lo, 1.0).astype(np.float32)
    return ((actions.astype(np.float32) + 1.0) * 0.5) * diff + lo.astype(np.float32)


def write_action_stats(meta_path: Path, lo: np.ndarray, hi: np.ndarray) -> None:
    """Add `action_normalization` block to an `info.json` file.

    Idempotent: if the block exists, we leave it alone so existing datasets
    aren't accidentally re-normalized on subsequent opens.
    """
    meta_path = Path(meta_path)
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)
    else:
        meta = {}
    if "action_normalization" in meta:
        return
    meta["action_normalization"] = {
        "lo": lo.astype(float).tolist(),
        "hi": hi.astype(float).tolist(),
        "note": ("Per-dim linear map [-pi_s..pi_s, -1..1] -> [-1, 1]. "
                 "Compute dataset-wide min/max at recording time."),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


def load_action_stats(meta_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read action_normalization block from info.json. Fall back to defaults."""
    meta_path = Path(meta_path)
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)
        stats = meta.get("action_normalization")
        if stats and "lo" in stats and "hi" in stats:
            return np.asarray(stats["lo"], dtype=np.float32), np.asarray(stats["hi"], dtype=np.float32)
    return DEFAULT_LO.copy(), DEFAULT_HI.copy()
