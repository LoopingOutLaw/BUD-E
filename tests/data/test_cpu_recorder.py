"""Tests for the CPU-only MuJoCo recorder (no JAX/GPU allocation)."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bude_vla.data.cpu_recorder import CPURecorder, record_dataset_cpu
from bude_vla.data.lerobot_v3 import BUDEDataset


def test_cpu_recorder_collect_reach():
    rec = CPURecorder()
    target = np.array([0.65, 0.05, 0.45], dtype=np.float32)
    ep = rec.collect_reach(target, n_steps=10)
    assert ep["images"].shape == (10, 64, 64, 3)
    assert ep["proprio"].shape == (10, 8)
    assert ep["actions"].shape[1] == 7
    assert ep["instruction"] == "reach the red target"


def test_cpu_recorder_collect_push():
    rec = CPURecorder()
    target_2d = np.array([0.05, -0.05], dtype=np.float32)
    ep = rec.collect_push(target_2d, n_steps=10)
    assert ep["images"].shape[0] <= 10
    assert ep["images"].shape[1:] == (64, 64, 3)
    assert ep["proprio"].shape[1] == 8
    assert ep["actions"].shape[1] == 7
    assert "push" in ep["instruction"].lower()


def test_cpu_recorder_no_jax_import():
    """Verify the module can be imported without JAX being used."""
    import importlib
    import sys
    mod_path = "bude_vla.data.cpu_recorder"
    if mod_path in sys.modules:
        del sys.modules[mod_path]
    mod = importlib.import_module(mod_path)
    assert not hasattr(mod, "jax") or "jax" not in dir(mod)


def test_record_dataset_cpu_creates_episodes():
    with tempfile.TemporaryDirectory() as td:
        record_dataset_cpu(task="reach", n_episodes=3, n_steps=5, root=td)
        ds = BUDEDataset(td)
        assert ds.num_episodes() == 3


def test_record_dataset_cpu_writes_videos_and_parquets():
    with tempfile.TemporaryDirectory() as td:
        record_dataset_cpu(task="reach", n_episodes=2, n_steps=4, root=td)
        root = Path(td)
        assert (root / "meta" / "info.json").exists()
        parquets = list(root.rglob("*.parquet"))
        mp4s = list(root.rglob("*.mp4"))
        assert len(parquets) == 2
        assert len(mp4s) == 4   # 2 episodes × 2 cameras (overhead + wrist)
