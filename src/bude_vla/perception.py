"""Lightweight image-derived perception features.

These helpers use only RGB pixels, not simulator state. They are therefore
available in both sim and real camera deployments when the object is visually
marked (red cube in this task).
"""
from __future__ import annotations

import numpy as np


def _current_top_rgb(image: np.ndarray, n_history_frames: int) -> np.ndarray | None:
    arr = np.asarray(image)
    if arr.ndim != 3:
        return None
    channels = arr.shape[-1]
    offset = max(0, (n_history_frames - 1) * 6) if channels >= n_history_frames * 6 else 0
    rgb = arr[..., offset:offset + 3]
    return rgb if rgb.shape[-1] == 3 else None


def detect_red_candidates(image: np.ndarray, n_history_frames: int = 1) -> np.ndarray:
    """Return red connected components as [x, y, score], best first.

    Treating all red pixels as one blob makes the centroid jump toward red arm
    parts during approach. Components preserve object identity so a runtime
    tracker can associate the cube over time.
    """
    rgb_u8 = _current_top_rgb(image, n_history_frames)
    if rgb_u8 is None:
        return np.empty((0, 3), dtype=np.float32)

    import cv2

    rgb = rgb_u8.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    contrast = r - np.maximum(g, b)
    mask = (
        (r > 65.0)
        & (contrast > 20.0)
        & (r > g * 1.22)
        & (r > b * 1.22)
    ).astype(np.uint8)

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    h, w = rgb.shape[:2]
    candidates: list[tuple[float, float, float]] = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        component = labels == label
        ys, xs = np.nonzero(component)
        weights = np.maximum(contrast[component], 1.0)
        cx = float(np.sum(xs * weights) / np.sum(weights))
        cy = float(np.sum(ys * weights) / np.sum(weights))
        width = max(1, int(stats[label, cv2.CC_STAT_WIDTH]))
        height = max(1, int(stats[label, cv2.CC_STAT_HEIGHT]))
        squareness = min(width, height) / max(width, height)
        fill = area / float(width * height)
        score = float(area * np.mean(weights) * (0.5 + squareness) * (0.5 + fill))
        x = (cx / max(1.0, w - 1.0)) * 2.0 - 1.0
        y = (cy / max(1.0, h - 1.0)) * 2.0 - 1.0
        candidates.append((x, y, score))

    candidates.sort(key=lambda item: item[2], reverse=True)
    if not candidates:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(candidates, dtype=np.float32)


def detect_red_centroid(image: np.ndarray, n_history_frames: int = 1) -> np.ndarray:
    """Return the strongest normalized red component as [x, y, valid]."""
    candidates = detect_red_candidates(image, n_history_frames=n_history_frames)
    if not len(candidates):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([candidates[0, 0], candidates[0, 1], 1.0], dtype=np.float32)
