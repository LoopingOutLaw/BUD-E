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
import torch

# ── Character-level tokenizer for instructions ──────────────────────────

_CHAR_VOCAB = {chr(i): i + 2 for i in range(32, 127)}  # printable ASCII -> 2..126
_CHAR_VOCAB["<pad>"] = 0
_CHAR_VOCAB["<unk>"] = 1
_TEXT_MAX_LEN = 64


def _tokenize_instruction(text: str, max_len: int = _TEXT_MAX_LEN) -> np.ndarray:
    """Convert instruction string to token IDs array of length max_len."""
    ids = [_CHAR_VOCAB.get(c, 1) for c in text.strip()[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


def _domain_from_instruction(text: str) -> int:
    """Heuristic: reach=0, push=1."""
    t = text.lower()
    if "push" in t:
        return 1
    return 0


META = {
    "fps": 30,
    "robot_type": "ur5e_so101_sim",
    "codebase_version": "v3.0",
    "features": {
        "observation.state": {"dtype": "float32", "shape": [8]},
        "action": {"dtype": "float32", "shape": [7]},
        "observation.images.top": {"dtype": "video", "shape": "auto"},
        "language_instruction": {"dtype": "string", "shape": [1]},
    },
}


def _frame_index_dir(root: Path) -> Path:
    return root / "meta" / "episodes_index"


def write_episode(root: str | Path, episode: dict) -> Path:
    """Write one episode to the v3 layout. Return path to the parquet file."""
    root = Path(root)
    images = episode["images"]                        # (T, H, W, 3) uint8
    proprio = episode["proprio"].astype(np.float32)     # (T, 8) arm+gripper
    actions = episode["actions"].astype(np.float32)   # (T, 7)
    instruction = episode["instruction"]

    T = images.shape[0]
    assert proprio.shape[0] == T, f"proprio length {proprio.shape[0]} != T {T}"
    assert proprio.shape[1] == 8, f"proprio dim {proprio.shape[1]} != 8"

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
        "observation.state": pa.array(list(proprio)),
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


class BUDETrainingDataset:
    """PyTorch Dataset for training BUD-E policy from LeRobot v3 data.

    On first .read(), decodes all MP4 frames into a single .npy file in the
    dataset root and memory-maps it. Subsequent reads skip the decode and
    just mmap the .npy — instant startup, zero MP4 decoding at training time.
    """

    def __init__(self, root: str | Path, chunk_size: int = 4):
        self.root = Path(root)
        self.chunk_size = chunk_size
        self._episodes: list[dict] = []
        self._cum_frames: list[int] = []
        self._total_frames = 0
        self._images: np.ndarray | None = None

    def read(self) -> "BUDETrainingDataset":
        from pyarrow import parquet as pq

        ep_index = _frame_index_dir(self.root)
        ep_files = sorted(ep_index.glob("*.json"))

        for ep_meta_path in ep_files:
            ep_meta = json.loads(ep_meta_path.read_text())
            ep_idx = ep_meta["episode_index"]
            chunk_idx = ep_idx // 1000
            base = self.root / "data" / f"chunk-{chunk_idx:03d}"

            pq_path = base / f"episode_{ep_idx:06d}.parquet"
            table = pq.read_table(str(pq_path))
            states = np.stack([np.asarray(row.as_py(), dtype=np.float32)
                               for row in table["observation.state"]])
            actions = np.stack([np.asarray(row.as_py(), dtype=np.float32)
                                for row in table["action"]])
            instruction = table["language_instruction"][0].as_py()
            T = states.shape[0]

            self._episodes.append({
                "states": states,
                "actions": actions,
                "instruction": instruction,
                "token_ids": _tokenize_instruction(instruction),
                "domain_id": _domain_from_instruction(instruction),
                "length": T,
                "ep_idx": ep_idx,
                "chunk_idx": chunk_idx,
            })
            self._cum_frames.append(self._total_frames)
            self._total_frames += T

        npy_path = self.root / "all_images.npy"
        if npy_path.exists():
            self._images = np.load(str(npy_path), mmap_mode="r")
            assert self._images.shape[0] == self._total_frames, \
                f"npy has {self._images.shape[0]} frames, expected {self._total_frames}"
        else:
            self._precache_images(npy_path)

        return self

    def _precache_images(self, npy_path: Path):
        import imageio.v3 as iio
        first_ep = self._episodes[0]
        vid0 = (self.root / "videos" / f"chunk-{first_ep['chunk_idx']:03d}" /
                "observation.images.top" / f"episode_{first_ep['ep_idx']:06d}.mp4")
        sample = iio.imread(str(vid0), plugin="pyav")
        H, W, C = sample.shape[1], sample.shape[2], sample.shape[3]
        all_imgs = np.zeros((self._total_frames, H, W, C), dtype=np.uint8)
        offset = 0
        for ep in self._episodes:
            chunk_idx = ep["chunk_idx"]
            ep_idx = ep["ep_idx"]
            vid_path = (self.root / "videos" / f"chunk-{chunk_idx:03d}" /
                        "observation.images.top" / f"episode_{ep_idx:06d}.mp4")
            frames = iio.imread(str(vid_path), plugin="pyav")
            T = frames.shape[0]
            all_imgs[offset:offset + T] = frames
            offset += T
        np.save(str(npy_path), all_imgs)
        self._images = np.load(str(npy_path), mmap_mode="r")

    def __len__(self) -> int:
        return self._total_frames

    def __getitem__(self, idx: int) -> dict:
        ep_i = 0
        while ep_i < len(self._cum_frames) - 1 and idx >= self._cum_frames[ep_i + 1]:
            ep_i += 1
        frame_in_ep = idx - self._cum_frames[ep_i]
        ep = self._episodes[ep_i]

        img = torch.from_numpy(
            self._images[idx].astype(np.float32)
        ).permute(2, 0, 1) / 255.0
        st = torch.from_numpy(ep["states"][frame_in_ep])
        txt = torch.from_numpy(ep["token_ids"])
        dom = torch.tensor(ep["domain_id"], dtype=torch.long)

        a_slice = ep["actions"][frame_in_ep: frame_in_ep + self.chunk_size]
        n = a_slice.shape[0]
        if n < self.chunk_size:
            pad = np.zeros((self.chunk_size - n, 7), dtype=np.float32)
            a_slice = np.concatenate([a_slice, pad], axis=0)
        act = torch.from_numpy(a_slice)

        tau = torch.rand(1).squeeze()
        noise = torch.randn_like(act)

        return {
            "images": img,
            "text_ids": txt,
            "proprio": st,
            "domain_id": dom,
            "actions": act,
            "tau": tau,
            "noise": noise,
        }
