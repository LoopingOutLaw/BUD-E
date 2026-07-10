"""Convert joint-target pick datasets to TCP-delta action datasets.

The learned policy must not receive cube coordinates. This script only rewrites
expert actions from absolute joint targets into observable end-effector motion
commands: [dx, dy, dz, gripper]. Images and proprio are preserved, videos are
symlinked by default to avoid duplicating tens of GB of MP4 data.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bude_vla.action_space import joint_action_to_ee_delta
from bude_vla.data.lerobot_v3 import finalize_dataset
from bude_vla.envs.so101_mjx import load_arm_model


def _rows_to_array(col) -> np.ndarray:
    return np.asarray(col.to_pylist(), dtype=np.float32)


def _copy_or_link(src: Path, dst: Path, *, copy_videos: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_videos:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def convert_root(src_root: Path, dst_root: Path, *, overwrite: bool,
                 max_delta: float, copy_videos: bool) -> None:
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_root} exists; pass --overwrite to replace it")
        shutil.rmtree(dst_root)

    ep_index = src_root / "meta" / "episodes_index"
    ep_files = sorted(ep_index.glob("*.json"))
    if not ep_files:
        raise FileNotFoundError(f"No episodes found under {ep_index}")

    model = load_arm_model()
    fk_data = mujoco.MjData(model)

    (dst_root / "meta" / "episodes_index").mkdir(parents=True, exist_ok=True)
    converted_frames = 0
    for n, ep_meta_path in enumerate(ep_files):
        ep_meta = json.loads(ep_meta_path.read_text())
        ep_idx = int(ep_meta["episode_index"])
        chunk_idx = ep_idx // 1000
        src_pq = src_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        table = pq.read_table(str(src_pq))
        states = _rows_to_array(table.column("observation.state"))
        actions = _rows_to_array(table.column("action"))
        if actions.shape[1] != 6:
            raise ValueError(f"{src_pq} action_dim={actions.shape[1]}, expected 6 joint action")
        if states.shape[1] < 6:
            raise ValueError(f"{src_pq} state_dim={states.shape[1]}, expected at least 6")

        ee_actions = np.stack([
            joint_action_to_ee_delta(model, fk_data, state, action, max_delta=max_delta)
            for state, action in zip(states, actions)
        ]).astype(np.float32)

        dst_chunk = dst_root / "data" / f"chunk-{chunk_idx:03d}"
        dst_chunk.mkdir(parents=True, exist_ok=True)
        dst_pq = dst_chunk / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(pa.table({
            "observation.state": table.column("observation.state"),
            "action": pa.array(list(ee_actions)),
            "language_instruction": table.column("language_instruction"),
        }), str(dst_pq))

        for cam in ("observation.images.top", "observation.images.wrist"):
            src_vid = src_root / "videos" / f"chunk-{chunk_idx:03d}" / cam / f"episode_{ep_idx:06d}.mp4"
            dst_vid = dst_root / "videos" / f"chunk-{chunk_idx:03d}" / cam / f"episode_{ep_idx:06d}.mp4"
            if src_vid.exists():
                _copy_or_link(src_vid, dst_vid, copy_videos=copy_videos)

        (dst_root / "meta" / "episodes_index" / ep_meta_path.name).write_text(json.dumps(ep_meta, indent=2))
        converted_frames += table.num_rows
        if n % 100 == 0:
            print(f"converted {n}/{len(ep_files)} episodes frames={converted_frames}", flush=True)

    src_info = src_root / "meta" / "info.json"
    info = json.loads(src_info.read_text()) if src_info.exists() else {}
    features = dict(info.get("features", {}))
    features["action"] = {"dtype": "float32", "shape": [4]}
    info["features"] = features
    info["action_space"] = "ee_delta"
    info["ee_delta_max_delta"] = float(max_delta)
    info.pop("action_normalization", None)
    (dst_root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    stats = finalize_dataset(dst_root)
    print(f"DONE {src_root} -> {dst_root} episodes={len(ep_files)} frames={converted_frames}")
    print(f"action_norm lo={stats['lo']} hi={stats['hi']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source joint-action dataset root")
    ap.add_argument("--out", required=True, help="Output ee_delta dataset root")
    ap.add_argument("--max-delta", type=float, default=0.08,
                    help="Clip converted TCP deltas to +/- this many meters before normalization")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--copy-videos", action="store_true",
                    help="Copy MP4s instead of symlinking. Uses much more disk.")
    args = ap.parse_args()
    convert_root(Path(args.src), Path(args.out), overwrite=args.overwrite,
                 max_delta=args.max_delta, copy_videos=args.copy_videos)


if __name__ == "__main__":
    main()
