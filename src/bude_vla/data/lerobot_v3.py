"""LeRobot Dataset v3 writer/reader.

v3 layout (matches `lerobot` v0.4+ on-disk format):
    meta/info.json                # fps, features, robot type
    data/chunk-XXX/episode_YYYYYY.parquet   # per-episode tabular state/action
    videos/chunk-XXX/observation.images.top/episode_YYYYYY.mp4  # video frames

This is a minimal implementation that supports what we need: write (image,
state, action, instruction) per episode, then read back via a torch-compatible
index. We deliberately avoid pulling in the full `lerobot` dataset class to
keep deps light and behavior explicit for BUD-E.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List

import numpy as np


META = {
    "fps": 30,
    "robot_type": "ur5e_so101_sim",
    "codebase_version": "v3.0",
    "features": {
        "observation.state": {"dtype": "float32", "shape": [8]},
        "action": {"dtype": "float32", "shape": [7]},
        "observation.images.top": {"dtype": "video", "shape": [64, 64, 3]},
        "language_instruction": {"dtype": "string", "shape": [1]},
    },
}


def _frame_index_dir(root: Path) -> Path:
    return root / "meta" / "episodes_index"


def write_episode(root: str | Path, episode: dict) -> Path:
    """Write one episode to the v3 layout. Return path to the parquet file."""
    root = Path(root)
    images = episode["images"]                        # (T, H, W, 3) uint8
    qpos = episode["qpos"].astype(np.float32)         # (T, 8)
    actions = episode["actions"].astype(np.float32)   # (T, 7)
    instruction = episode["instruction"]

    T = images.shape[0]
    assert qpos.shape[0] == T, f"qpos length {qpos.shape[0]} != T {T}"
    assert actions.shape[0] == T

    # Find next episode_idx
    episodes_index = _frame_index_dir(root)
    episodes_index.mkdir(parents=True, exist_ok=True)
    existing = sorted(episodes_index.glob("*.json"))
    episode_idx = len(existing)

    chunk_idx = episode_idx // 1000
    chunk_dir = root / "data" / f"chunk-{chunk_idx:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # ---- Write MP4 video ----
    import imageio
    vid_dir = root / "videos" / f"chunk-{chunk_idx:03d}" / "observation.images.top"
    vid_dir.mkdir(parents=True, exist_ok=True)
    vid_path = vid_dir / f"episode_{episode_idx:06d}.mp4"
    writer = imageio.get_writer(str(vid_path), fps=META["fps"], codec="libx264", quality=8)
    for frame in images:
        writer.append_data(frame)
    writer.close()

    # ---- Write parquet (tabular state/action/instruction) ----
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Per-frame instruction: encode the same string for every frame
    instr_arr = np.array([instruction] * T, dtype=object)

    table = pa.table({
        "observation.state": pa.array(list(qpos)),
        "action": pa.array(list(actions)),
        "language_instruction": pa.array(instr_arr, type=pa.string()),
    })
    pq_path = chunk_dir / f"episode_{episode_idx:06d}.parquet"
    pq.write_table(table, str(pq_path))

    # ---- Write info.json and episodes_index ----
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(META, indent=2))

    ep_meta = {
        "episode_index": episode_idx,
        "tasks": [instruction],
        "length": T,
    }
    (episodes_index / f"episode_{episode_idx:06d}.json").write_text(json.dumps(ep_meta, indent=2))
    return pq_path


class BUDEDataset:
    """Minimal lerobot-v3 style reader. Loads parquet(s) into memory at __init__."""

    def __init__(self, root: str | Path):
        from pyarrow import parquet as pq
        self.root = Path(root)
        episodes_index = _frame_index_dir(self.root)
        if not episodes_index.exists():
            raise FileNotFoundError(f"No meta/episodes_index in {self.root}")
        self.episode_files = sorted(episodes_index.glob("*.json"))
        if not self.episode_files:
            raise FileNotFoundError(f"No episode_*.json in {episodes_index}")

        # Read all parquets (small datasets expected for sim)
        self.frames: List[np.ndarray] = []
        for ep_meta_path in self.episode_files:
            ep_meta = json.loads(ep_meta_path.read_text())
            ep_idx = ep_meta["episode_index"]
            chunk_idx = ep_idx // 1000
            pq_path = self.root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
            table = pq.read_table(str(pq_path))
            self.frames.append(table)

    def __len__(self) -> int:
        return sum(t.num_rows for t in self.frames)

    def get_episode(self, idx: int) -> dict:
        """Return {state, action, instruction} for a single episode (idx across episodes)."""
        cum = 0
        for t in self.frames:
            n = t.num_rows
            if idx < cum + n:
                local = idx - cum
                return {
                    "state": np.stack([np.asarray(t["observation.state"][i].as_py()) for i in range(local, n)]),
                    "action": np.stack([np.asarray(t["action"][i].as_py()) for i in range(local, n)]),
                    "instruction": t["language_instruction"][local].as_py(),
                }
            cum += n
        raise IndexError(idx)

    def num_episodes(self) -> int:
        return len(self.frames)
