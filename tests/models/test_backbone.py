"""Tests for the transformer backbone."""
import torch
from bude_vla.models.backbone import PolicyTransformer


def test_backbone_output_shape_preserved():
    m = PolicyTransformer(d=64, depth=2, heads=4, ffn_dim=128)
    x = torch.randn(2, 100, 64)
    out = m(x)
    assert out.shape == (2, 100, 64)


def test_backbone_does_not_collapse():
    torch.manual_seed(0)
    m = PolicyTransformer(d=32, depth=2, heads=2, ffn_dim=64)
    m.eval()
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        out = m(x)
    # Different positions should produce different outputs
    assert out.std(dim=1).mean().item() > 1e-3


def test_backbone_freeze_first_n():
    m = PolicyTransformer(d=32, depth=4, heads=2, ffn_dim=64)
    m.freeze_first_n(2)
    # First two blocks' params should be frozen
    frozen = [not p.requires_grad for blk in m.blocks[:2] for p in blk.parameters()]
    trainable = [p.requires_grad for blk in m.blocks[2:] for p in blk.parameters()]
    assert all(frozen)
    assert all(trainable)
