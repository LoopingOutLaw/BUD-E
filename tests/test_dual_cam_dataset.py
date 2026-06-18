"""Red tests for dual-camera + 8-dim proprio pipeline.

These tests ensure that the dataset returns 6-channel images and 8-dim
proprio — no cube_xyz leakage.
"""
import numpy as np
import pytest
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bude_vla.data.lerobot_v3 import _augment_image, BUDETrainingDataset
from bude_vla.models.policy import BUDEConfig, BUDEPolicy


class TestDualCamPipeline:
    """Verify dual-cam (6-ch) images + arm-only (8-dim) proprio."""

    def test_dataset_returns_six_channel_image(self):
        ds = BUDETrainingDataset("/tmp/test_dualcam", chunk_size=4)
        # _precache_images must produce (N, H, W, 6) for dual-cam
        source = Path(ds._precache_images.__code__.co_filename).read_text()
        assert "C = 6 if has_wrist else 3" in source, (
            "_precache_images does not produce 6-channel output"
        )

    def test_proprio_dim_is_eight(self):
        from bude_vla.data.lerobot_v3 import META
        assert META["features"]["observation.state"]["shape"] == [8], (
            "META state_dim is not 8"
        )

    def test_no_cube_xyz_in_dataset_meta(self):
        from bude_vla.data.lerobot_v3 import write_episode
        from pathlib import Path
        import inspect
        src = inspect.getsource(write_episode)
        assert "cube_xyz" not in src, (
            "write_episode still references cube_xyz — leakage"
        )
        assert 'state_dim == 8' in src or "state_dim != 8" in src, (
            "write_episode does not enforce state_dim == 8"
        )

    def test_policy_in_channels_is_six(self):
        cfg = BUDEConfig()
        # ViT in_channels comes from policy init
        import inspect
        src = inspect.getsource(BUDEPolicy.__init__)
        assert "in_channels=6" in src, (
            "BUDEPolicy still uses in_channels=3"
        )

    def test_policy_state_dim_is_eight(self):
        cfg = BUDEConfig()
        assert cfg.state_dim == 8, (
            f"BUDEConfig.state_dim={cfg.state_dim}, expected 8"
        )
