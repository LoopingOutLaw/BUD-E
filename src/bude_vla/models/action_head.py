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
    """Predicts velocity field given (noisy_action, time tau, conditioning)."""

    def __init__(self, action_dim: int = 7, chunk_size: int = 32, d: int = 256,
                 time_dim: int = 128, hidden_dim: int = 512, n_steps: int = 10):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_steps = n_steps

        # conditioning projector: from backbone hidden state (one row) -> hidden_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # time + action MLP
        self.time_dim = time_dim
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_in_proj = nn.Linear(action_dim, hidden_dim)
        self.mlp = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, action_dim),
            ) for _ in range(chunk_size)]
        )
        # The block above indexes per token rather than per chunk; instead, use a
        # shared 4-layer MLP operating on per-token features. We replace it:
        self.mlp = None  # placeholder for clarity

        # Actually use a shared MLP applied per token after time+cond+action fusion.
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, noisy_action: torch.Tensor, tau: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """
        noisy_action: (B, chunk_size, action_dim) — x_t
        tau: (B,) — timesteps in [0,1]
        cond: (B, d) — conditioning vector from backbone (e.g., state token)

        Returns: (B, chunk_size, action_dim) predicted velocity field.
        """
        b, t, a = noisy_action.shape
        tau_emb = sinusoidal_time_embedding(tau, self.time_dim)  # (B, time_dim)
        tau_proj = self.time_proj(tau_emb)                       # (B, hidden_dim)
        cond_proj = self.cond_proj(cond)                         # (B, hidden_dim)

        # Broadcast: (B, 1, hidden_dim) added to per-token action projection
        act_proj = self.action_in_proj(noisy_action)             # (B, T, hidden_dim)
        h = act_proj + tau_proj.unsqueeze(1) + cond_proj.unsqueeze(1)
        v = self.shared_mlp(h)                                   # (B, T, action_dim)
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
        return x
