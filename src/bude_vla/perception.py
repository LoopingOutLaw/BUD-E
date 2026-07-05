"""Lightweight image-derived perception features.

These helpers use only RGB pixels, not simulator state. They are therefore
available in both sim and real camera deployments when the object is visually
marked (red cube in this task).
"""
from __future__ import annotations

import numpy as np


def detect_red_centroid(image: np.ndarray, n_history_frames: int = 1) -> np.ndarray:
    """Return normalized red-object centroid [x, y, valid].

    `image` may be a single dual-cam frame (H,W,6) or a history-stacked frame
    (H,W,n_history_frames*6). The detector uses the current top-camera RGB.
    x/y are normalized to [-1, 1] in image coordinates.
    """
    arr = np.asarray(image)
    if arr.ndim != 3:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    c = arr.shape[-1]
    if c >= n_history_frames * 6:
        off = max(0, (n_history_frames - 1) * 6)
    else:
        off = 0
    rgb = arr[..., off:off + 3].astype(np.float32)
    if rgb.shape[-1] < 3:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r > 80.0) & (r > g * 1.4) & (r > b * 1.4)
    weights = None

    if int(mask.sum()) < 4:
        # MP4-compressed cache frames can desaturate the cube enough that the
        # ratio mask misses it. Fall back to the strongest red-contrast pixels,
        # which remain image-derived and deployable on real camera frames.
        score = np.maximum(r - np.maximum(g, b), r - 0.5 * (g + b))
        thr = max(18.0, float(np.percentile(score, 99.85)))
        mask = (r > 70.0) & (score >= thr)
        if int(mask.sum()) < 4:
            thr = max(12.0, float(np.percentile(score, 99.5)))
            mask = (r > 60.0) & (score >= thr)
        weights = np.maximum(score[mask] - thr + 1.0, 1.0) if int(mask.sum()) >= 4 else None

    ys, xs = np.nonzero(mask)
    if xs.size < 4:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    h, w = rgb.shape[:2]
    if weights is not None and weights.shape[0] == xs.shape[0]:
        cx = float((xs.astype(np.float32) * weights).sum() / weights.sum())
        cy = float((ys.astype(np.float32) * weights).sum() / weights.sum())
    else:
        cx = float(xs.mean())
        cy = float(ys.mean())
    x = (cx / max(1.0, w - 1.0)) * 2.0 - 1.0
    y = (cy / max(1.0, h - 1.0)) * 2.0 - 1.0
    return np.array([x, y, 1.0], dtype=np.float32)
