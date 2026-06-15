"""Tests for the full assembled BUD-E policy."""
import torch
from bude_vla.models.policy import BUDEPolicy, BUDEConfig


def _make_batch(cfg: BUDEConfig, B: int = 2):
    return {
        "images": torch.randn(B, 3, cfg.img_size, cfg.img_size),
        "text_ids": torch.randint(1, cfg.text_vocab, (B, cfg.text_max_len)),
        "proprio": torch.randn(B, cfg.state_dim),
        "domain_id": torch.zeros(B, dtype=torch.long),
        "tau": torch.rand(B),
        "noise": torch.randn(B, cfg.chunk_size, cfg.action_dim),
        "actions": torch.randn(B, cfg.chunk_size, cfg.action_dim),
    }


def test_policy_forward_shape():
    cfg = BUDEConfig(img_size=32, vision_dim=32, vision_depth=1, vision_heads=2,
                     text_max_len=8, text_depth=1, d=64, backbone_depth=1,
                     backbone_heads=2, state_dim=8, action_dim=7, chunk_size=4,
                     n_domains=2, text_vocab=64, action_head_hidden_dim=64,
                     action_head_time_dim=32, ffn_dim=64)
    p = BUDEPolicy(cfg)
    b = _make_batch(cfg, B=2)
    out = p(b)
    assert "velocity" in out
    assert out["velocity"].shape == (2, 4, 7)


def test_policy_backbone_token_count():
    cfg = BUDEConfig(img_size=32, d=64, backbone_depth=1, backbone_heads=2,
                     n_domains=2, n_prompts=8, text_max_len=4, action_head_hidden_dim=64,
                     action_head_time_dim=32, ffn_dim=64)
    p = BUDEPolicy(cfg)
    b = _make_batch(cfg, B=1)
    b["text_ids"] = torch.randint(1, cfg.text_vocab, (1, 3))  # shorter text
    out = p(b)
    # backbone tokens: n_prompts(8) + state(1) + patches((32/16)**2=4) + text(3) = 16
    # we don't expose this directly but we check tokens shape
    assert out["tokens"].shape[1] == 8 + 1 + 4 + 3


def test_policy_sample_shape():
    cfg = BUDEConfig(img_size=32, vision_dim=32, vision_depth=1, vision_heads=2,
                     text_max_len=8, text_depth=1, d=64, backbone_depth=1,
                     backbone_heads=2, state_dim=8, action_dim=7, chunk_size=4,
                     n_domains=1, text_vocab=64, action_head_hidden_dim=64,
                     action_head_time_dim=32, ffn_dim=64)
    p = BUDEPolicy(cfg)
    p.eval()
    batch = {
        "images": torch.randn(2, 3, 32, 32),
        "text_ids": torch.randint(1, cfg.text_vocab, (2, 8)),
        "proprio": torch.randn(2, 8),
        "domain_id": torch.zeros(2, dtype=torch.long),
    }
    with torch.no_grad():
        actions = p.sample(batch)
    assert actions.shape == (2, 4, 7)


def test_policy_total_params_under_30M():
    """Production-shaped total should still fit comfortably under 30M (allow overhead)."""
    cfg = BUDEConfig()  # defaults
    p = BUDEPolicy(cfg)
    counts = p.n_params()
    assert counts["total"] < 30_000_000, f"Policy too big: {counts}"
    assert counts["total"] > 5_000_000, f"Policy too small: {counts}"


def test_policy_soft_prompts_separate_per_domain():
    """Different domain ids should produce different backbone internal states."""
    torch.manual_seed(42)
    cfg = BUDEConfig(img_size=32, vision_dim=32, vision_depth=1, vision_heads=2,
                     text_max_len=8, text_depth=1, d=64, backbone_depth=1,
                     backbone_heads=2, action_head_hidden_dim=64,
                     action_head_time_dim=32, ffn_dim=64, n_domains=2,
                     n_prompts=8, chunk_size=4)
    p = BUDEPolicy(cfg)
    p.eval()
    base = {
        "images": torch.randn(1, 3, 32, 32),
        "text_ids": torch.randint(1, cfg.text_vocab, (1, 8)),
        "proprio": torch.randn(1, cfg.state_dim),
        "actions": torch.randn(1, cfg.chunk_size, cfg.action_dim),
        "tau": torch.rand(1),
        "noise": torch.randn(1, cfg.chunk_size, cfg.action_dim),
    }
    with torch.no_grad():
        batch0 = {**base, "domain_id": torch.zeros(1, dtype=torch.long)}
        batch1 = {**base, "domain_id": torch.ones(1, dtype=torch.long)}
        out0 = p(batch0)
        out1 = p(batch1)
    assert not torch.allclose(out0["tokens"], out1["tokens"], atol=1e-6), \
        "Soft prompts had no effect on backbone outputs at all"
    n_p = cfg.n_prompts
    s0 = out0["tokens"][:, n_p, :]
    s1 = out1["tokens"][:, n_p, :]
    assert not torch.allclose(s0, s1, atol=1e-6), \
        "Soft prompts didn't propagate to state token"
