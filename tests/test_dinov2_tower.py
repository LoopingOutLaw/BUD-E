"""Tests for DINOv2Tower: forward shape, CLS stripping, and frozen blocks."""
import pytest
import torch

from bude_vla.models.vision import DINOv2Tower


@pytest.fixture
def tower():
    return DINOv2Tower(img_size=224, in_channels=6, out_dim=256, finetune_blocks=4)


def test_forward_output_shape(tower):
    x = torch.randn(2, 6, 224, 224)
    out = tower(x)
    n_patches = tower.backbone.patch_embed.num_patches
    assert out.shape == (2, n_patches, 256), f"got {out.shape}, expected (2, {n_patches}, 256)"


def test_no_cls_in_output(tower):
    x = torch.randn(1, 6, 224, 224)
    out = tower(x)
    backbone_patches = tower.backbone.patch_embed.num_patches
    assert out.shape[1] == backbone_patches, (
        f"output has {out.shape[1]} tokens but backbone has {backbone_patches} patches"
    )


def test_early_blocks_frozen(tower):
    total = len(tower.backbone.blocks)
    for i, block in enumerate(tower.backbone.blocks):
        for p in block.parameters():
            if i < total - 4:
                assert not p.requires_grad, f"block {i} should be frozen"
            else:
                assert p.requires_grad, f"block {i} should be trainable"


def test_norm_and_pos_embed_trainable(tower):
    for p in tower.backbone.norm.parameters():
        assert p.requires_grad
    assert tower.backbone.pos_embed.requires_grad


def test_patch_embed_6ch(tower):
    assert tower.backbone.patch_embed.proj.in_channels == 6


def test_first_3ch_weights_preserved(tower):
    w = tower.backbone.patch_embed.proj.weight
    assert w.shape[1] == 6
    assert w[:, 3:].abs().max() == 0.0, "extra channels should be zero-initialized"
