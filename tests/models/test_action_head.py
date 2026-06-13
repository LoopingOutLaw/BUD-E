"""Tests for the flow-matching action head."""
import torch
from bude_vla.models.action_head import FlowMatchingActionHead, sinusoidal_time_embedding


def test_sinusoidal_embedding_shape():
    t = torch.tensor([0.0, 0.5, 1.0])
    emb = sinusoidal_time_embedding(t, dim=128)
    assert emb.shape == (3, 128)


def test_sinusoidal_embedding_periodic():
    """sin should oscillate at various frequencies."""
    t = torch.linspace(0, 1, 100)
    emb = sinusoidal_time_embedding(t, dim=32)
    # First dim should be sin at lowest frequency (shouldn't be constant)
    assert emb[:, 0].std() > 0.1


def test_action_head_training_forward_shape():
    h = FlowMatchingActionHead(action_dim=7, chunk_size=8, d=64,
                               time_dim=32, hidden_dim=128, n_steps=10)
    b = 2
    x = torch.randn(b, 8, 7)
    tau = torch.rand(b)
    cond = torch.randn(b, 64)
    v = h(x, tau, cond)
    assert v.shape == (b, 8, 7)


def test_action_head_inference_sample_shape():
    h = FlowMatchingActionHead(action_dim=4, chunk_size=5, d=32,
                               time_dim=16, hidden_dim=64, n_steps=5)
    b = 3
    cond = torch.randn(b, 32)
    a = h.sample(cond)
    assert a.shape == (b, 5, 4)


def test_action_head_bounded_param_count():
    h = FlowMatchingActionHead(action_dim=7, chunk_size=32, d=256,
                               time_dim=128, hidden_dim=512, n_steps=10)
    n = sum(p.numel() for p in h.parameters())
    assert n < 6_000_000, f"Action head too big: {n}"
