"""Tests for batch dataset recording into LeRobot v3 format."""
import tempfile
from pathlib import Path

import numpy as np

from bude_vla.data.cpu_recorder import record_dataset_cpu
from bude_vla.data.lerobot_v3 import BUDEDataset


def test_record_dataset_creates_reach_episodes():
    with tempfile.TemporaryDirectory() as td:
        record_dataset_cpu(task="reach", n_episodes=3, n_steps=5, root=td)
        ds = BUDEDataset(td)
        assert ds.num_episodes() == 3
        assert len(ds) >= 3


def test_record_dataset_writes_videos_and_parquets():
    with tempfile.TemporaryDirectory() as td:
        record_dataset_cpu(task="reach", n_episodes=2, n_steps=4, root=td)
        root = Path(td)
        assert (root / "meta" / "info.json").exists()
        parquets = list(root.rglob("*.parquet"))
        mp4s = list(root.rglob("*.mp4"))
        assert len(parquets) == 2
        assert len(mp4s) == 2


def test_record_dataset_push_task():
    with tempfile.TemporaryDirectory() as td:
        record_dataset_cpu(task="push", n_episodes=2, n_steps=5, root=td)
        ds = BUDEDataset(td)
        assert ds.num_episodes() == 2
