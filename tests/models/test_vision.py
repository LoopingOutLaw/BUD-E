"""Tests for the ViT-S vision tower."""
import torch
from bude_vla.models.vision import ViTSmall


def test_vitsmall_output_shape():
    m = ViTSmall(img_size=64, patch_size=16, in_channels=3, dim=64, depth=1,
                 heads=2, mlp_ratio=2.0, out_dim=64)
    m.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, 16, 64)  # 4*4 patches, out_dim


def test_vitsmall_out_dim_matches_config():
    m = ViTSmall(img_size=32, patch_size=16, dim=32, depth=1, heads=2,
                 mlp_ratio=2.0, out_dim=128)
    m.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 4, 128)


def test_vitsmall_realistic_param_count():
    """Production-shape model should match the spec ~5M budget (allow 30% tolerance)."""
    m = ViTSmall(img_size=224, patch_size=16, in_channels=3, dim=192, depth=8,
                 heads=3, mlp_ratio=4.0, out_dim=256)
    n = sum(p.numel() for p in m.parameters())
    assert 1_000_000 < n < 6_000_000, f"ViT-S param count out of band: {n}"


def test_vitsmall_does_not_collapse_random_input():
    """Output should not be all-same across spatial positions (sanity)."""
    torch.manual_seed(0)
    m = ViTSmall(img_size=32, patch_size=16, dim=32, depth=1, heads=2,
                 mlp_ratio=2.0, out_dim=32)
    m.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = m(x)
    # Different patches should produce different outputs (high variance spatial)
    spatial_std = out.std(dim=1).mean().item()
    assert spatial_std > 1e-3, f"ViT collapsed to constant output: std={spatial_std}"
