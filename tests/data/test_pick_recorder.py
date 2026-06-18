"""Tests for pick_recorder."""
import tempfile
from pathlib import Path
from bude_vla.data.pick_recorder import record_pick_episode


def test_record_pick_episode_returns_correct_arrays():
    with tempfile.TemporaryDirectory() as td:
        ep = record_pick_episode(root=td, episode_idx=0, cube_xy=(0.6, 0.0))
        assert "images" in ep
        assert "proprio" in ep
        assert "actions" in ep
        assert "instruction" in ep
        assert ep["proprio"].shape[1] == 8
        assert ep["actions"].shape[1] == 7
        assert "pick" in ep["instruction"].lower()


def test_record_pick_episode_returns_success_flag():
    with tempfile.TemporaryDirectory() as td:
        ep = record_pick_episode(root=td, episode_idx=0, cube_xy=(0.6, 0.0))
        assert "success" in ep
        assert isinstance(ep["success"], bool)


def test_record_pick_episode_randomizes_cube_position():
    with tempfile.TemporaryDirectory() as td:
        ep1 = record_pick_episode(root=td, episode_idx=0, cube_xy=(0.55, 0.05))
        ep2 = record_pick_episode(root=td, episode_idx=1, cube_xy=(0.70, -0.10))
        assert ep1["images"].shape[0] != 0
        assert ep2["images"].shape[0] != 0
        assert not (
            (ep1["proprio"][0] == ep2["proprio"][0]).all()
            and (ep1["proprio"][-1] == ep2["proprio"][-1]).all()
        )
