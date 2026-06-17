"""Tests for P0+P1 architecture upgrades:
- MiniLMTextEncoder (frozen pretrained + tiny trainable proj)
- FlowMatchingActionHead with optional causal temporal transformer
- BUDETrainingDataset with n_history_frames stacking
- BUDEConfig wiring (use_dinov2, use_minilm, n_history_frames)
"""
import pytest
import torch
from bude_vla.models.action_head import FlowMatchingActionHead
from bude_vla.models.policy import BUDEConfig, BUDEPolicy


# ────────────── MiniLMTextEncoder ──────────────

def test_minilm_text_encoder_shapes():
    """MiniLMTextEncoder outputs (B, max_len, d)."""
    from bude_vla.models.text_encoder import MiniLMTextEncoder
    enc = MiniLMTextEncoder(d=256, max_len=32)
    enc = enc.to("cuda" if torch.cuda.is_available() else "cpu")
    out = enc(["pick the red cube", "push the blue block"])
    assert out.shape == (2, 32, 256), f"got {out.shape}"


def test_minilm_backbone_frozen():
    """All MiniLM backbone params must have requires_grad=False."""
    from bude_vla.models.text_encoder import MiniLMTextEncoder
    enc = MiniLMTextEncoder(d=256, max_len=32)
    n_backbone = sum(1 for _ in enc.model.parameters())
    n_train = sum(1 for p in enc.model.parameters() if p.requires_grad)
    assert n_backbone > 0
    assert n_train == 0, f"expected 0 trainable backbone params, got {n_train}"


def test_minilm_projection_only_trainable():
    """Only the proj layer should be trainable in MiniLMTextEncoder."""
    from bude_vla.models.text_encoder import MiniLMTextEncoder
    enc = MiniLMTextEncoder(d=256, max_len=32)
    trainable_params = [n for n, _ in enc.named_parameters() if _.requires_grad]
    assert trainable_params == ["proj.weight", "proj.bias"], trainable_params


# ────────────── Causal Temporal Action Head ──────────────

def test_flow_head_no_temporal_legacy():
    """temporal_depth=0 must keep the legacy per-token behavior."""
    head = FlowMatchingActionHead(action_dim=7, chunk_size=4,
                                  temporal_depth=0)
    x_t = torch.randn(2, 4, 7)
    tau = torch.rand(2)
    cond = torch.randn(2, 256)
    v = head(x_t, tau, cond)
    assert v.shape == (2, 4, 7)
    assert not head.temporal_depth


def test_flow_head_with_temporal():
    """temporal_depth=2 adds a causal transformer; output shape unchanged."""
    head = FlowMatchingActionHead(action_dim=7, chunk_size=8,
                                  temporal_depth=2, temporal_heads=4)
    x_t = torch.randn(2, 8, 7)
    tau = torch.rand(2)
    cond = torch.randn(2, 256)
    v = head(x_t, tau, cond)
    assert v.shape == (2, 8, 7)
    assert head.temporal_depth == 2
    assert hasattr(head, "temporal")
    assert hasattr(head, "causal_mask")
    assert head.causal_mask.shape == (64, 64)


def test_flow_head_sample_with_temporal():
    """Inference denoise loop must work with temporal layers on."""
    head = FlowMatchingActionHead(action_dim=7, chunk_size=4,
                                  temporal_depth=2, n_steps=5)
    cond = torch.randn(2, 256)
    sample = head.sample(cond)
    assert sample.shape == (2, 4, 7)


# ────────────── BUDEConfig + BUDEPolicy wiring ──────────────

def test_policy_default_config():
    """Default: TinyTextEncoder + ViTSmall + Markovian. ~19M params."""
    cfg = BUDEConfig()
    policy = BUDEPolicy(cfg)
    parts = policy.n_params()
    assert parts["total"] < 25_000_000, f"got {parts['total']}"


def test_policy_with_all_features():
    """use_dinov2 + use_minilm + n_history_frames=2 should compose cleanly."""
    cfg = BUDEConfig(use_dinov2=True, use_minilm=True, n_history_frames=2)
    policy = BUDEPolicy(cfg)
    assert policy.text.__class__.__name__ == "MiniLMTextEncoder"
    assert policy.vision.__class__.__name__ == "DINOv2Tower"
    parts = policy.n_params()
    assert parts["text"] > 1_000_000, "MiniLM should add ~22M params"


def test_policy_in_channels_scales_with_history():
    """n_history_frames=2 should double the in_channels of the vision tower."""
    cfg = BUDEConfig()  # default n_history=1, in_channels=6
    cfg1 = BUDEConfig(n_history_frames=1)
    cfg2 = BUDEConfig(n_history_frames=2)
    p1 = BUDEPolicy(cfg1)
    p2 = BUDEPolicy(cfg2)
    in_ch_1 = p1.vision.patch_embed.proj.in_channels
    in_ch_2 = p2.vision.patch_embed.proj.in_channels
    assert in_ch_1 == 6
    assert in_ch_2 == 12, f"expected 12, got {in_ch_2}"


def test_policy_dinov2_with_history():
    """DINOv2 backbone must accept 12-channel input when history=2."""
    cfg = BUDEConfig(use_dinov2=True, n_history_frames=2)
    policy = BUDEPolicy(cfg)
    images = torch.randn(2, 12, 224, 224)
    tokens = policy.vision(images)
    assert tokens.shape[1] == 256  # 256 patches at 224 with patch14
    assert tokens.shape[-1] == 256  # out_dim
