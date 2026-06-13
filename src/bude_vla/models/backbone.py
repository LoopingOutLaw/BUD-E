"""8-layer Transformer encoder backbone for BUD-E.

Input: [soft_prompts(32) | state_token(1) | patch_tokens(196) | text_tokens(T_text)].
Output: same-length sequence; downstream head indexes the state position for action queries.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-LN standard transformer block."""

    def __init__(self, d: int, heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class PolicyTransformer(nn.Module):
    """Stack of N transformer blocks. Returns the final hidden states."""

    def __init__(self, d: int = 256, depth: int = 8, heads: int = 8, ffn_dim: int = 1024,
                 dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.depth = depth
        self.heads = heads
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, heads, ffn_dim, dropout) for _ in range(depth)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            tokens = blk(tokens)
        return tokens

    def freeze_first_n(self, n: int) -> None:
        """Freeze the first n transformer blocks (used in Phase II)."""
        for i, blk in enumerate(self.blocks):
            requires = i >= n
            for p in blk.parameters():
                p.requires_grad = requires
