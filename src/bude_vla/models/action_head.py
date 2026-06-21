"""Flow-matching action head (OT-CFM).

Predicts the velocity field v(x_t, tau) where x_t = (1-tau) * x0 + tau * x1.
Trained with MSE between predicted v and ground-truth v = x1 - x0.
Inference: Euler integration from x_0 ~ N(0, I) over T denoising steps.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """Map (B,) timesteps in [0,1] to (B, dim) sinusoid embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(1, half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[1]))
    return emb


class FlowMatchingActionHead(nn.Module):
    """Predicts velocity field given (noisy_action, time tau, conditioning).

    P1: when temporal_depth > 0, applies a causal transformer over the
    per-timestep MLP outputs so each timestep can attend to earlier
    timesteps in the action chunk. This makes trajectories smooth.
    """

    def __init__(self, action_dim: int = 7, chunk_size: int = 32, d: int = 256,
                 time_dim: int = 128, hidden_dim: int = 512, n_steps: int = 10,
                 temporal_depth: int = 2, temporal_heads: int = 4,
                 max_temporal_len: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_steps = n_steps
        self.temporal_depth = temporal_depth

        self.cond_proj = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.time_dim = time_dim
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_in_proj = nn.Linear(action_dim, hidden_dim)

        # Per-token shared MLP applied across the timestep dimension
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        if temporal_depth > 0:
            # Project action_dim -> hidden, run causal transformer, project back
            self.temporal_in = nn.Linear(action_dim, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=temporal_heads,
                dim_feedforward=hidden_dim * 2, activation="gelu",
                dropout=0.0, batch_first=True, norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(
                layer, num_layers=temporal_depth,
            )
            self.temporal_out = nn.Linear(hidden_dim, action_dim)
            causal = torch.triu(
                torch.ones(max_temporal_len, max_temporal_len, dtype=torch.bool),
                diagonal=1,
            )
            self.register_buffer("causal_mask", causal)

    def _apply_temporal(self, v: torch.Tensor) -> torch.Tensor:
        if self.temporal_depth == 0:
            return v
        T = v.shape[1]
        h = self.temporal_in(v)
        h = self.temporal(h, mask=self.causal_mask[:T, :T])
        h = self.temporal_out(h)
        return v + h

    def forward(self, noisy_action: torch.Tensor, tau: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """
        noisy_action: (B, chunk_size, action_dim) — x_t
        tau: (B,) — timesteps in [0,1]
        cond: (B, d) — conditioning vector from backbone (e.g., state token)

        Returns: (B, chunk_size, action_dim) predicted velocity field.
        """
        b, t, a = noisy_action.shape
        tau_emb = sinusoidal_time_embedding(tau, self.time_dim)
        tau_proj = self.time_proj(tau_emb)
        cond_proj = self.cond_proj(cond)

        act_proj = self.action_in_proj(noisy_action)
        h = act_proj + tau_proj.unsqueeze(1) + cond_proj.unsqueeze(1)
        v = self.shared_mlp(h)
        v = self._apply_temporal(v)
        return v

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Euler denoise from x_0 ~ N(0, I) to predicted x_1 (the action chunk)."""
        device = cond.device
        b = cond.shape[0]
        x = torch.randn(b, self.chunk_size, self.action_dim, device=device,
                        generator=generator)
        for step in range(self.n_steps):
            tau = torch.full((b,), step / self.n_steps, device=device)
            v = self.forward(x, tau, cond)
            x = x + v / self.n_steps
        return torch.clamp(x, -1.0, 1.0)
