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

from bude_vla.perception import detect_red_centroid
from bude_vla.data.action_normalization import (
    DEFAULT_HI,
    DEFAULT_LO,
    compute_action_stats,
    denormalize_actions,
    load_action_stats,
    normalize_actions,
    pad_scale,
    write_action_stats,
)

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
    """Heuristic: reach=0, push=1, pick=2."""
    t = text.lower()
    if "push" in t:
        return 1
    if "pick" in t:
        return 2
    return 0


def _augment_image(img_chw: torch.Tensor, rng: np.random.Generator,
                   brightness_range: float = 0.10,
                   crop_pad: int = 8) -> torch.Tensor:
    """Augment an image tensor (C, H, W float32 in [0, 1]) with random crop
    and brightness jitter. Returns same shape.

    Crop is reflective-padding by `crop_pad` then cropping back. Pads are
    small relative to 224×224 — keeps the cube and target zone mostly inside
    the crop while giving 55% pixel translation variance. Brightness jitter
    adds a uniform scalar in [-brightness_range, +brightness_range].",
    """
    c, h, w = img_chw.shape
    pad = int(crop_pad)
    if pad > 0:
        padded = torch.nn.functional.pad(
            img_chw.unsqueeze(0), (pad, pad, pad, pad), mode="reflect"
        ).squeeze(0)
        dy = int(rng.integers(0, 2 * pad + 1))
        dx = int(rng.integers(0, 2 * pad + 1))
        img_chw = padded[:, dy:dy + h, dx:dx + w]
    if brightness_range > 0.0:
        delta = float(rng.uniform(-brightness_range, brightness_range))
        img_chw = (img_chw + delta).clamp(0.0, 1.0)
    return img_chw


META = {
    "fps": 30,
    "robot_type": "ur5e_so101_sim",
    "codebase_version": "v3.1",
    "features": {
        "observation.state": {"dtype": "float32", "shape": [6]},
        "action": {"dtype": "float32", "shape": [6]},
        "observation.images.top": {"dtype": "video", "shape": "auto"},
        "observation.images.wrist": {"dtype": "video", "shape": "auto"},
        "language_instruction": {"dtype": "string", "shape": [1]},
    },
}


def _frame_index_dir(root: Path) -> Path:
    return root / "meta" / "episodes_index"


def _proprio_dim(episode: dict) -> int:
    """Return proprio dim from episode dict (6 for arm+gripper)."""
    return int(episode["proprio"].shape[1])


def write_episode(root: str | Path, episode: dict) -> Path:
    """Write one episode to the v3 layout. Return path to the parquet file.

    Action values are stored in the natural (un-normalized) range. After
    all episodes are recorded, `finalize_dataset(root)` should be called to
    compute per-dim min/max stats and persist them in `meta/info.json` under
    `action_normalization`. The training path then loads and applies this
    transform at __getitem__ time so the flow head works in [-1, 1].
    """
    root = Path(root)
    images = episode["images"]                        # (T, H, W, 3) uint8
    proprio = episode["proprio"].astype(np.float32)     # (T, state_dim)
    actions = episode["actions"].astype(np.float32)   # (T, action_dim)
    instruction = episode["instruction"]

    T = images.shape[0]
    assert proprio.shape[0] == T, f"proprio length {proprio.shape[0]} != T {T}"
    state_dim = int(proprio.shape[1])
    action_dim = int(actions.shape[1])
    assert state_dim in (6, 7, 9), (
        f"proprio dim {state_dim} not in (6, 7, 9)"
    )

    # Find next episode_idx
    episodes_index = _frame_index_dir(root)
    episodes_index.mkdir(parents=True, exist_ok=True)
    existing = sorted(episodes_index.glob("*.json"))
    episode_idx = len(existing)

    chunk_idx = episode_idx // 1000
    chunk_dir = root / "data" / f"chunk-{chunk_idx:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # ---- Write dual-cam MP4 videos (split 6-ch → two 3-ch streams) ----
    import imageio
    overhead_dir = root / "videos" / f"chunk-{chunk_idx:03d}" / "observation.images.top"
    wrist_dir = root / "videos" / f"chunk-{chunk_idx:03d}" / "observation.images.wrist"
    overhead_dir.mkdir(parents=True, exist_ok=True)
    wrist_dir.mkdir(parents=True, exist_ok=True)

    overhead_path = overhead_dir / f"episode_{episode_idx:06d}.mp4"
    wrist_path = wrist_dir / f"episode_{episode_idx:06d}.mp4"

    if images.shape[-1] == 6:
        split_overhead = images[:, :, :, :3]
        split_wrist = images[:, :, :, 3:]
    elif images.shape[-1] == 3:
        split_overhead = images
        split_wrist = images
    else:
        raise ValueError(f"images has {images.shape[-1]} channels, expected 3 or 6")

    for vid_path, vid_frames in [(overhead_path, split_overhead), (wrist_path, split_wrist)]:
        writer = imageio.get_writer(str(vid_path), fps=META["fps"],
                                    codec="libx264", quality=8)
        for frame in vid_frames:
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
        local_meta = {**META,
                      "features": {**META["features"],
                                   "observation.state": {
                                       "dtype": "float32",
                                       "shape": [state_dim],
                                   },
                                   "action": {
                                       "dtype": "float32",
                                       "shape": [action_dim],
                                   }}}
        info_path.write_text(json.dumps(local_meta, indent=2))

    ep_meta = {
        "episode_index": episode_idx,
        "tasks": [instruction],
        "length": T,
    }
    (episodes_index / f"episode_{episode_idx:06d}.json").write_text(json.dumps(ep_meta, indent=2))
    return pq_path


def finalize_dataset(root: str | Path, action_margin: float = 0.02) -> dict:
    from pyarrow import parquet as pq

    root = Path(root)
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        existing = json.loads(info_path.read_text())
        if "action_normalization" in existing:
            return existing["action_normalization"]

    episodes_index = _frame_index_dir(root)
    ep_files = sorted(episodes_index.glob("*.json"))
    if not ep_files:
        raise FileNotFoundError(f"No episodes in {root}")

    def _iter_actions():
        for ep_meta_path in ep_files:
            ep_meta = json.loads(ep_meta_path.read_text())
            ep_idx = ep_meta["episode_index"]
            chunk_idx = ep_idx // 1000
            pq_path = root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
            table = pq.read_table(str(pq_path))
            actions = np.stack([np.asarray(row.as_py(), dtype=np.float32)
                                for row in table["action"]])
            yield actions

    lo, hi = compute_action_stats(_iter_actions())
    lo_p, hi_p = np.zeros_like(lo), np.zeros_like(hi)
    for d in range(lo.shape[0]):
        lo_p[d], hi_p[d] = pad_scale(float(lo[d]), float(hi[d]), margin=action_margin)

    write_action_stats(info_path, lo_p, hi_p)

    return {
        "lo": lo_p.astype(float).tolist(),
        "hi": hi_p.astype(float).tolist(),
    }


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

    def __init__(self, root: str | Path, chunk_size: int = 4,
                 augment: bool = False,
                 brightness_range: float = 0.10,
                 crop_pad: int = 8,
                 normalize: bool = True,
                 n_history_frames: int = 1,
                 lazy_videos: bool = True,
                 lazy_cache_size: int = 8,
                 frame_cache: str | Path | None = None,
                 action_stats: tuple[np.ndarray, np.ndarray] | None = None):
        self.root = Path(root)
        self.chunk_size = chunk_size
        self.augment = augment
        self.brightness_range = brightness_range
        self.crop_pad = crop_pad
        self.normalize = normalize
        self.n_history_frames = n_history_frames
        # When True, skip the giant all_images.npy precache and decode MP4s
        # lazily per-episode. Required when the .npy would not fit on disk
        # (1.5 TiB cache on a 338 GB volume, for example) or when the cache
        # is corrupted.
        self._lazy_videos = lazy_videos
        self._lazy_cache_size = lazy_cache_size
        # Per-worker LRU cache of decoded episodes: ep_idx -> (T,H,W,6) uint8.
        # Insertion order = LRU order; pop(0) evicts oldest.
        self._lazy_cache: dict[int, np.ndarray] = {}
        self._frame_cache_dir = Path(frame_cache) if frame_cache is not None else None
        self._cache_images: np.ndarray | None = None
        self._cache_global_indices: np.ndarray | None = None
        self._episodes: list[dict] = []
        self._cum_frames: list[int] = []
        self._total_frames = 0
        self._images: np.ndarray | None = None
        self._has_wrist: bool = True
        self._rng = np.random.default_rng()
        self._action_lo: np.ndarray | None = None
        self._action_hi: np.ndarray | None = None
        self._action_stats_override = action_stats

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

        if self._frame_cache_dir is not None:
            img_path = self._frame_cache_dir / "images.uint8.npy"
            idx_path = self._frame_cache_dir / "global_indices.npy"
            if not img_path.exists() or not idx_path.exists():
                raise FileNotFoundError(
                    f"frame cache missing {img_path} or {idx_path}; run scripts/build_frame_cache.py first")
            self._cache_images = np.load(str(img_path), mmap_mode="r")
            self._cache_global_indices = np.load(str(idx_path), mmap_mode="r")
            if self._cache_images.shape[0] != self._cache_global_indices.shape[0]:
                raise ValueError("frame cache image/index length mismatch")
            self._images = None
            self._total_frames = int(self._cache_global_indices.shape[0])
            if self.normalize:
                if self._action_stats_override is not None:
                    self._action_lo, self._action_hi = self._action_stats_override
                else:
                    info_path = self.root / "meta" / "info.json"
                    self._action_lo, self._action_hi = load_action_stats(info_path)
            return self

        npy_path = self.root / "all_images.npy"
        use_npy = False
        if not self._lazy_videos and npy_path.exists():
            existing = np.load(str(npy_path), mmap_mode="r")
            if existing.shape[0] == self._total_frames:
                # Sanity: at least the first episode must have real content.
                if existing.ndim == 4 and np.asarray(existing[0]).max() > 0:
                    self._images = existing
                    self._has_wrist = existing.shape[-1] == 6
                    use_npy = True
                else:
                    import warnings
                    warnings.warn(
                        f"all_images.npy exists but its first frame is all-zero "
                        f"(corrupted cache). Falling back to lazy MP4 loading.")
            else:
                import warnings
                warnings.warn(
                    f"all_images.npy shape mismatch: {existing.shape[0]} vs "
                    f"{self._total_frames} expected. Falling back to lazy MP4 loading.")
        if not use_npy:
            self._images = None  # lazy mode: decode per-episode via _load_episode_frames

        if self.normalize:
            info_path = self.root / "meta" / "info.json"
            if self._action_stats_override is not None:
                self._action_lo, self._action_hi = self._action_stats_override
            else:
                self._action_lo, self._action_hi = load_action_stats(info_path)
                try:
                    _is_default = (self._action_lo.shape == DEFAULT_LO.shape
                                   and (self._action_lo == DEFAULT_LO).all()
                                   and (self._action_hi == DEFAULT_HI).all())
                except (ValueError, AttributeError):
                    _is_default = False
                if _is_default:
                    import warnings
                    warnings.warn(
                        f"action_normalization missing in {info_path}; falling back "
                        f"to defaults. Call finalize_dataset() after recording to fix."
                    )

        return self

    def _precache_images(self, npy_path: Path):
        import imageio.v3 as iio
        if not self._episodes:
            raise RuntimeError(
                f"No episodes loaded from {self.root}. "
                f"Check that the dataset path is correct and contains episode files.")
        first_ep = self._episodes[0]
        chunk0 = first_ep["chunk_idx"]
        ep0 = first_ep["ep_idx"]
        vid_top0 = (self.root / "videos" / f"chunk-{chunk0:03d}" /
                    "observation.images.top" / f"episode_{ep0:06d}.mp4")
        vid_wrist0 = (self.root / "videos" / f"chunk-{chunk0:03d}" /
                      "observation.images.wrist" / f"episode_{ep0:06d}.mp4")
        sample_top = iio.imread(str(vid_top0), plugin="pyav")
        H, W = sample_top.shape[1], sample_top.shape[2]
        has_wrist = vid_wrist0.exists()
        C = 6 if has_wrist else 3
        all_imgs = np.lib.format.open_memmap(
            str(npy_path), mode="w+",
            dtype=np.uint8, shape=(self._total_frames, H, W, C))
        offset = 0
        for ep in self._episodes:
            chunk_idx = ep["chunk_idx"]
            ep_idx = ep["ep_idx"]
            vid_top = (self.root / "videos" / f"chunk-{chunk_idx:03d}" /
                       "observation.images.top" / f"episode_{ep_idx:06d}.mp4")
            frames_top = iio.imread(str(vid_top), plugin="pyav")
            T = frames_top.shape[0]
            if has_wrist:
                vid_wrist = (self.root / "videos" / f"chunk-{chunk_idx:03d}" /
                             "observation.images.wrist" / f"episode_{ep_idx:06d}.mp4")
                frames_wrist = iio.imread(str(vid_wrist), plugin="pyav")
                all_imgs[offset:offset + T] = np.concatenate(
                    [frames_top, frames_wrist], axis=-1)
            else:
                all_imgs[offset:offset + T, :, :, :3] = frames_top
                all_imgs[offset:offset + T, :, :, 3:] = frames_top
            offset += T
        all_imgs.flush()
        del all_imgs
        self._images = np.load(str(npy_path), mmap_mode="r")

    def _load_episode_frames(self, ep_idx: int) -> np.ndarray:
        cached = self._lazy_cache.get(ep_idx)
        if cached is not None:
            return cached
        ep = next(e for e in self._episodes if e["ep_idx"] == ep_idx)
        chunk_idx = ep["chunk_idx"]
        vid_top = (self.root / "videos" / f"chunk-{chunk_idx:03d}" /
                   "observation.images.top" / f"episode_{ep_idx:06d}.mp4")
        import imageio.v3 as iio
        frames_top = iio.imread(str(vid_top), plugin="pyav")
        vid_wrist = (self.root / "videos" / f"chunk-{chunk_idx:03d}" /
                     "observation.images.wrist" / f"episode_{ep_idx:06d}.mp4")
        if vid_wrist.exists():
            frames_wrist = iio.imread(str(vid_wrist), plugin="pyav")
            frames = np.concatenate([frames_top, frames_wrist], axis=-1)
        else:
            frames = np.concatenate([frames_top, frames_top], axis=-1)
        self._lazy_cache[ep_idx] = frames
        while len(self._lazy_cache) > self._lazy_cache_size:
            self._lazy_cache.pop(next(iter(self._lazy_cache)))
        return frames

    def __len__(self) -> int:
        return self._total_frames

    def __getitem__(self, idx: int) -> dict:
        cache_row = None
        if self._cache_global_indices is not None:
            cache_row = int(idx)
            idx = int(self._cache_global_indices[cache_row])

        ep_i = 0
        while ep_i < len(self._cum_frames) - 1 and idx >= self._cum_frames[ep_i + 1]:
            ep_i += 1
        frame_in_ep = idx - self._cum_frames[ep_i]
        ep = self._episodes[ep_i]

        # Stack n_history_frames: current frame plus up to n_history_frames-1
        # earlier frames from the same episode. Clamped to episode start.
        n_h = self.n_history_frames
        if self._cache_images is not None:
            stacked = self._cache_images[cache_row]
            expected_c = n_h * 6
            if stacked.shape[-1] != expected_c:
                raise ValueError(
                    f"frame cache has {stacked.shape[-1]} channels but "
                    f"n_history_frames={n_h} requires {expected_c}. "
                    "Rebuild the cache with matching --n-history-frames."
                )
        elif self._images is not None:
            if n_h <= 1:
                stacked = self._images[idx]
            else:
                start = max(0, frame_in_ep - (n_h - 1))
                window = self._images[idx - (frame_in_ep - start): idx + 1]
                if window.shape[0] < n_h:
                    pad_n = n_h - window.shape[0]
                    pad = np.repeat(window[:1], pad_n, axis=0)
                    window = np.concatenate([pad, window], axis=0)
                window = np.ascontiguousarray(window)
                stacked = np.transpose(window, (1, 2, 0, 3)).reshape(
                    window.shape[1], window.shape[2], window.shape[0] * window.shape[-1])
        else:
            ep_frames = self._load_episode_frames(ep["ep_idx"])
            T = ep_frames.shape[0]
            if frame_in_ep >= T:
                frame_in_ep = T - 1
            if n_h <= 1:
                stacked = ep_frames[frame_in_ep]
            else:
                start = max(0, frame_in_ep - (n_h - 1))
                sel = ep_frames[start: frame_in_ep + 1]
                if sel.shape[0] < n_h:
                    pad_n = n_h - sel.shape[0]
                    pad = np.repeat(sel[:1], pad_n, axis=0)
                    sel = np.concatenate([pad, sel], axis=0)
                sel = np.ascontiguousarray(sel)
                stacked = np.transpose(sel, (1, 2, 0, 3)).reshape(
                    sel.shape[1], sel.shape[2], sel.shape[0] * sel.shape[-1])

        perception = torch.from_numpy(detect_red_centroid(stacked, n_history_frames=n_h))
        img = torch.from_numpy(
            stacked.astype(np.float32)
        ).permute(2, 0, 1).contiguous() / 255.0
        if self.augment:
            img = _augment_image(img, self._rng,
                                 brightness_range=self.brightness_range,
                                 crop_pad=self.crop_pad)
        st = torch.from_numpy(ep["states"][frame_in_ep])
        txt = torch.from_numpy(ep["token_ids"])
        dom = torch.tensor(ep["domain_id"], dtype=torch.long)

        a_slice = ep["actions"][frame_in_ep: frame_in_ep + self.chunk_size]
        n = a_slice.shape[0]
        mask = torch.ones(self.chunk_size, dtype=torch.float32)
        if n < self.chunk_size:
            action_dim = a_slice.shape[-1] if a_slice.ndim == 2 else 6
            pad = np.zeros((self.chunk_size - n, action_dim), dtype=np.float32)
            a_slice = np.concatenate([a_slice, pad], axis=0)
            mask[n:] = 0.0
        if self.normalize and self._action_lo is not None:
            a_slice = normalize_actions(a_slice, self._action_lo, self._action_hi)
        act = torch.from_numpy(a_slice.astype(np.float32))

        tau = torch.rand(1).squeeze()
        noise = torch.randn_like(act)

        phase = torch.tensor(
            float(frame_in_ep) / float(max(1, ep["length"] - 1)),
            dtype=torch.float32,
        )

        return {
            "images": img,
            "text_ids": txt,
            "instruction": ep["instruction"],
            "proprio": st,
            "perception": perception,
            "domain_id": dom,
            "actions": act,
            "tau": tau,
            "noise": noise,
            "mask": mask,
            "phase": phase,
        }
