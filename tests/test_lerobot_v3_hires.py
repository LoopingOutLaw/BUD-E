"""Verify lerobot_v3 precaches the right shape (not hardcoded 64x64)."""
import inspect
from bude_vla.data.lerobot_v3 import BUDETrainingDataset


def test_precache_supports_224():
    ds = BUDETrainingDataset("/tmp/test_lerobot_hires", chunk_size=4)
    source = inspect.getsource(ds._precache_images)
    assert "64, 64, 3" not in source, (
        "_precache_images still hardcodes 64x64"
    )


def test_meta_shape_is_not_fixed_dim():
    from bude_vla.data import lerobot_v3 as lv3
    src = inspect.getsource(lv3)
    assert '"shape": [64, 64, 3]' not in src, "META shape still hardcoded"
