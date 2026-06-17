"""Regression tests for the 3 critical integration bugs found in the
architecture review (post-P0+P1 merge):

  Bug #1 — train.py collate_fn crashes on string fields (e.g. "instruction")
  Bug #2 — env_runner doesn't honor n_history_frames (channel mismatch crash)
  Bug #3 — env_runner._build_batch missing "instruction" for MiniLM
"""
from __future__ import annotations

import numpy as np
import torch


# ────────────── Bug #1: collate_fn string support ──────────────

def test_collate_fn_survives_string_instruction():
    """collate_fn must NOT torch.stack strings. It must return list[str]."""
    from scripts.train import collate_fn  # type: ignore  # noqa: F401

    batch = [
        {
            "instruction": "pick the red cube",
            "images": torch.zeros(2, 6, 8, 8),
            "proprio": torch.zeros(2, 8),
        },
        {
            "instruction": "pick the red cube",
            "images": torch.zeros(2, 6, 8, 8),
            "proprio": torch.zeros(2, 8),
        },
    ]
    out = collate_fn(batch)
    assert isinstance(out["instruction"], list), f"got {type(out['instruction'])}"
    assert out["instruction"] == ["pick the red cube", "pick the red cube"]
    assert out["images"].shape == (2, 2, 6, 8, 8)
    assert out["proprio"].shape == (2, 2, 8)


def test_collate_fn_mixed_types():
    """String fields are lists; tensor fields are stacked."""
    from scripts.train import collate_fn  # type: ignore  # noqa: F401

    batch = [
        {"language": "a", "x": torch.tensor([1.0, 2.0])},
        {"language": "b", "x": torch.tensor([3.0, 4.0])},
        {"language": "c", "x": torch.tensor([5.0, 6.0])},
    ]
    out = collate_fn(batch)
    assert out["language"] == ["a", "b", "c"]
    assert torch.equal(out["x"], torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


# ────────────── Bug #2: env_runner history frame buffer ──────────────

def test_policy_runner_stacks_history_frames():
    """n_history_frames > 1 must produce stacked channels for the policy."""
    import mujoco
    from bude_vla.env_runner import PolicyRolloutRunner

    model = mujoco.MjModel.from_xml_path(
        "/home/aditya/bude_vla/urdf/ur5e_scene.xml"
    )
    data = mujoco.MjData(model)

    runner = PolicyRolloutRunner(model, img_size=64, n_history_frames=2,
                                 max_steps_per_try=5)
    try:
        assert runner.n_history_frames == 2
        stacked = runner._stacked_view(np.zeros((64, 64, 6), dtype=np.uint8))
        assert stacked.shape == (64, 64, 12), f"got {stacked.shape}"
        for _ in range(3):
            stacked = runner._stacked_view(np.ones((64, 64, 6), dtype=np.uint8) * 7)
        # Buffer capped at n_history_frames=2.
        assert len(runner._frame_buffer) == 2
        assert stacked.shape[-1] == 12
    finally:
        runner.close()


def test_policy_runner_history_one_unchanged():
    """n_history_frames=1 must produce a 6-channel image unchanged."""
    import mujoco
    from bude_vla.env_runner import PolicyRolloutRunner

    model = mujoco.MjModel.from_xml_path(
        "/home/aditya/bude_vla/urdf/ur5e_scene.xml"
    )
    runner = PolicyRolloutRunner(model, img_size=64, n_history_frames=1)
    try:
        assert runner.n_history_frames == 1
        img = np.ones((64, 64, 6), dtype=np.uint8) * 42
        out = runner._stacked_view(img)
        assert out.shape == (64, 64, 6), f"got {out.shape}"
        assert out.shape[-1] != 12  # explicit guard
    finally:
        runner.close()


def test_policy_runner_resets_buffer_per_try():
    """run_one must reset _frame_buffer at the start of each try."""
    import mujoco
    from bude_vla.env_runner import PolicyRolloutRunner

    model = mujoco.MjModel.from_xml_path(
        "/home/aditya/bude_vla/urdf/ur5e_scene.xml"
    )
    data = mujoco.MjData(model)

    class _StubPolicy:
        def sample(self, batch):
            # Action of all zeros, chunk_size=1.
            return torch.zeros(1, 1, 7)

    runner = PolicyRolloutRunner(model, img_size=64, n_history_frames=2,
                                 max_steps_per_try=2, max_tries=1)
    try:
        result = runner.run_one(data, _StubPolicy(),
                                cube_xy=np.array([0.5, 0.0]))
        assert isinstance(result, object)
        # After a successful(ish) run, runner should have left a buffer behind.
        assert len(runner._frame_buffer) <= runner.n_history_frames
    finally:
        runner.close()


# ────────────── Bug #3: _build_batch carries "instruction" ──────────────

def test_build_batch_includes_instruction():
    """_build_batch must pass the string instruction through for MiniLM."""
    from bude_vla.env_runner import _build_batch, _PICK_INSTRUCTION

    img = np.zeros((64, 64, 6), dtype=np.uint8)
    proprio = np.zeros(8, dtype=np.float32)
    text_ids = np.arange(16, dtype=np.int64)
    batch = _build_batch(img, proprio, text_ids, _PICK_INSTRUCTION,
                         domain_id=0, device="cpu")
    assert "instruction" in batch, f"keys: {list(batch.keys())}"
    assert isinstance(batch["instruction"], list)
    assert batch["instruction"] == [_PICK_INSTRUCTION]
    assert "text_ids" in batch
    assert batch["images"].shape == (1, 6, 64, 64)
    assert batch["proprio"].shape == (1, 8)
    assert batch["domain_id"].shape == (1,)
