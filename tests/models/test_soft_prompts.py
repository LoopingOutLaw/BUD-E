"""Tests for soft prompts."""
import torch
from bude_vla.models.soft_prompts import SoftPrompts, orthogonal_init_


def test_soft_prompts_single_domain():
    sp = SoftPrompts(n_domains=1, n_prompts=32, d=64)
    out = sp(0)
    assert out.shape == (32, 64)


def test_soft_prompts_gather_batch():
    sp = SoftPrompts(n_domains=6, n_prompts=32, d=64)
    domain_ids = torch.tensor([0, 3, 5, 0, 2, 1])
    out = sp.gather(domain_ids)
    assert out.shape == (6, 32, 64)
    assert torch.allclose(out[0], out[3])


def test_soft_prompts_different_domains_different_rows():
    sp = SoftPrompts(n_domains=6, n_prompts=8, d=32)
    a = sp(0)
    b = sp(1)
    assert not torch.allclose(a, b)


def test_soft_prompts_per_domain_is_orthogonal():
    """Each per-domain prompt block (N_p, d) should have roughly orthogonal rows.

    With N_p = d this is a square tested by Q Q^T ~= Identity.
    """
    sp = SoftPrompts(n_domains=2, n_prompts=8, d=8)
    for i in range(2):
        rows = sp(0)
        gram = rows @ rows.T
        # Check: diagonal constant, off-diagonal small
        diag_mean = gram.diag().mean()
        diag_std = gram.diag().std()
        off_diag_abs = gram - torch.diag(gram.diag())
        assert diag_std < 1e-3 * diag_mean, f"Diagonal not constant on domain {i}"
        assert off_diag_abs.abs().mean() < 0.1 * diag_mean, \
            f"Not orthogonal on domain {i}: {off_diag_abs.abs().mean()}"


def test_soft_prompts_param_count_5_domains():
    sp = SoftPrompts(n_domains=6, n_prompts=32, d=256)
    n = sum(p.numel() for p in sp.parameters())
    assert n < 100_000, f"Soft prompts too big: {n}"


def test_orthogonal_init_produces_constant_diag():
    """Test the standalone helper on a non-square (n<d) case."""
    t = torch.empty(6, 32)
    orthogonal_init_(t)
    # After QR on mxn with m<n, Q has orthonormal rows -> Q Q^T = I_m
    gram = t @ t.T
    target = torch.eye(6)
    assert torch.allclose(gram, target, atol=1e-5), \
        f"Orthogonal init did not produce orthonormal rows: max err {(gram - target).abs().max()}"
