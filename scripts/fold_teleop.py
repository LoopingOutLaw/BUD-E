"""Fold auto-recorded teleop episodes into LeRobot v3 layout.

Reads <root>/raw/*.npz + *.json (produced by teleop_demo.py), re-encodes as
parquet + mp4 chunks, and writes meta/episodes_index entries so BUDEDataset /
BUDETrainingDataset can read them.

Usage:
  PYTHONPATH=src /home/aditya/.bude-venv/bin/python scripts/fold_teleop.py \
    --root /home/aditya/bude_vla/data/teleop_v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bude_vla.data.lerobot_v3 import write_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/aditya/bude_vla/data/teleop_v3")
    args = ap.parse_args()
    root = Path(args.root)
    raw_dir = root / "raw"
    assert raw_dir.exists(), f"No raw/ dir at {raw_dir}"

    files = sorted(raw_dir.glob("episode_*.npz"))
    print(f"Folding {len(files)} raw npz into v3 layout under {root}")
    done = 0
    for npz_path in files:
        with np.load(npz_path, allow_pickle=True) as d:
            ep = {
                "instruction": str(d["instruction"].item()) if d["instruction"].ndim == 0
                              else str(d["instruction"][0]),
                "images": d["images"],
                "proprio": d["qpos"][:, 7:15].astype(np.float32),
                "actions": d["actions"].astype(np.float32),
            }
        if ep["proprio"].shape[1] != 8:
            print(f"  skip {npz_path.name}: bad proprio shape {ep['proprio'].shape}")
            continue
        try:
            write_episode(root, ep)
            done += 1
        except Exception as e:
            print(f"  fail {npz_path.name}: {e}")
    print(f"  -> wrote {done} episodes to v3 layout at {root}")


if __name__ == "__main__":
    main()
