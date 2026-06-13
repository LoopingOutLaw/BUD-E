"""Proprioception projector: 7D joint state -> d-dim token."""
from __future__ import annotations

import torch
import torch.nn as nn


class ProprioProjector(nn.Module):
    """Single linear: (state_dim -> d), with LayerNorm + SiLU for stability."""

    def __init__(self, state_dim: int = 7, out_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.out_dim = out_dim
        self.norm = nn.LayerNorm(state_dim)
        self.proj = nn.Linear(state_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.proj(x)
        return x
