"""Learnable soft-prompt table: (N_domains, N_p, d).

Each row is the embodiment/task embedding for one domain. Prepended to the
transformer backbone input to encode *which* robot + task the policy is acting on.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def orthogonal_init_(tensor: torch.Tensor) -> torch.Tensor:
    """Initialize with orthonormal rows via QR on random gaussian.

    Works for any (n, d) with n <= d. If n > d, falls back to plain orthogonal
    on the d rows and zero-pads to reach d cols.
    """
    n, d = tensor.shape
    if n <= d:
        # QR of a random nxd matrix gives Q with orthonormal columns, but we want
        # orthonormal rows. Use: M = randn(d, n), QR -> Q with orthonormal columns
        # -> Q.T has orthonormal rows of shape (n, d).
        m = torch.randn(d, n)
        q, _ = torch.linalg.qr(m)
        out = q[:, :n].t().contiguous()  # (n, d)
    else:
        # n > d: Gram-Schmidt on n random d-dim vectors, but only d of them can be
        # orthogonal. Fill remaining rows with scaled orthonormal residual.
        m = torch.randn(d, n)
        q, _ = torch.linalg.qr(m)
        out = q.t().contiguous()  # (n, d) but first d rows are orthonormal, rest random
    tensor.data.copy_(out)
    return tensor


class SoftPrompts(nn.Module):
    """Table of (N_domains, N_p, d) learnable prompt embeddings.

    Each domain's prompt block (N_p, d) is initialized with orthonormal rows
    so the domains start with disjoint representations.
    """

    def __init__(self, n_domains: int, n_prompts: int = 32, d: int = 256):
        super().__init__()
        self.n_domains = n_domains
        self.n_prompts = n_prompts
        self.d = d
        prompts = torch.empty(n_domains, n_prompts, d)
        for i in range(n_domains):
            orthogonal_init_(prompts[i])
        self.prompts = nn.Parameter(prompts)

    def forward(self, domain_id: int) -> torch.Tensor:
        return self.prompts[domain_id]

    def gather(self, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.prompts[domain_ids]
