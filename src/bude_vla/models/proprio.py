"""Proprioception projector: 7D joint state -> d-dim token."""
from __future__ import annotations

import torch
import torch.nn as nn


class PerFeatureAffine(nn.Module):
    """Learnable per-feature scaling without cross-feature information loss.

    LayerNorm is unsuitable for small physical vectors such as joint state or
    ``[pixel_x, pixel_y, valid]`` because it removes each sample's mean and
    scale across features. Distinct robot/object configurations can therefore
    become identical before the first learned projection. These inputs are
    already bounded, so an affine transform is sufficient and stays
    information preserving when its scale is non-zero.

    ``weight`` and ``bias`` intentionally match LayerNorm's state-dict keys and
    shapes. This lets a corrected policy initialize from an older checkpoint
    while changing only the forward semantics.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


def feature_transform(num_features: int, mode: str) -> nn.Module:
    if mode == "layernorm":
        return nn.LayerNorm(num_features)
    if mode == "affine":
        return PerFeatureAffine(num_features)
    raise ValueError(f"Unsupported input feature transform: {mode!r}")


class ProprioProjector(nn.Module):
    """Project deployable robot state into the policy token dimension."""

    def __init__(self, state_dim: int = 7, out_dim: int = 256,
                 feature_norm: str = "layernorm"):
        super().__init__()
        self.state_dim = state_dim
        self.out_dim = out_dim
        self.feature_norm = feature_norm
        self.norm = feature_transform(state_dim, feature_norm)
        self.proj = nn.Linear(state_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.proj(x)
        return x
