"""Tests for the proprio projector."""
import torch
from bude_vla.models.proprio import ProprioProjector


def test_proprio_shape():
    p = ProprioProjector(state_dim=7, out_dim=256)
    x = torch.randn(4, 7)
    out = p(x)
    assert out.shape == (4, 256)


def test_proprio_different_state_dim():
    p = ProprioProjector(state_dim=10, out_dim=64)
    x = torch.randn(2, 10)
    out = p(x)
    assert out.shape == (2, 64)


def test_proprio_param_count():
    """Should be tiny — well under 100k params."""
    p = ProprioProjector(state_dim=7, out_dim=256)
    n = sum(p.numel() for p in p.parameters())
    assert n < 100_000, f"Proprio too big: {n}"
