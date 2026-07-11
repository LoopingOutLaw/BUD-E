"""Vision towers for BUD-E.

ViTSmall       - from-scratch ViT-S (original, for backward compat).
DINOv2Tower    - pretrained DINOv2-small backbone with 6-channel adapter.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, dim: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSmall(nn.Module):
    """From-scratch ViT-S. No pretrained weights.

    Output: (B, num_patches, out_dim)
    Default production: out_dim=256, depth=12, dim=384 ~5M params.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        out_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim) if out_dim != dim else nn.Identity()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.proj(x)
        return x


def current_top_rgb_start(in_channels: int) -> int:
    """Return the channel offset of the current top-camera RGB frame."""
    if in_channels == 3:
        return 0
    if in_channels < 6 or in_channels % 6 != 0:
        raise ValueError(
            "DINOv2 input channels must be RGB or history-stacked dual-camera RGB"
        )
    return in_channels - 6


class DINOv2Tower(nn.Module):
    """Pretrained DINOv2-small with history-stacked dual-camera input.

    The pretrained RGB kernel is attached to the current top-camera channels;
    all history and wrist channels start at zero and are learned during
    fine-tuning. Every RGB group receives the ImageNet normalization expected
    by the frozen DINOv2 blocks.
    """

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 6,
        out_dim: int = 256,
        finetune_blocks: int = 4,
        pretrained: str = "vit_small_patch14_dinov2.lvd142m",
    ):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            pretrained, pretrained=True, img_size=img_size,
        )
        self.embed_dim = self.backbone.embed_dim
        self.num_patches = self.backbone.patch_embed.num_patches
        self.out_dim = out_dim

        self._adapt_patch_embed(in_channels)
        self._strip_cls_token()
        self._freeze_blocks(finetune_blocks)

        n_rgb_groups = in_channels // 3
        mean = torch.tensor([0.485, 0.456, 0.406] * n_rgb_groups).view(
            1, in_channels, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225] * n_rgb_groups).view(
            1, in_channels, 1, 1
        )
        self.register_buffer("input_mean", mean, persistent=False)
        self.register_buffer("input_std", std, persistent=False)
        self.proj = nn.Linear(self.embed_dim, out_dim) if out_dim != self.embed_dim else nn.Identity()

    def _adapt_patch_embed(self, in_channels: int) -> None:
        old_proj = self.backbone.patch_embed.proj
        if old_proj.in_channels == in_channels:
            return
        if old_proj.in_channels != 3:
            raise ValueError("expected a three-channel pretrained DINOv2 patch embed")
        new_proj = nn.Conv2d(
            in_channels, old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            bias=old_proj.bias is not None,
        )
        current_top = current_top_rgb_start(in_channels)
        with torch.no_grad():
            nn.init.zeros_(new_proj.weight)
            new_proj.weight[:, current_top:current_top + 3] = old_proj.weight
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        self.backbone.patch_embed.proj = new_proj

    def _strip_cls_token(self) -> None:
        pass

    def _freeze_blocks(self, finetune_blocks: int) -> None:
        total = len(self.backbone.blocks)
        for i, block in enumerate(self.backbone.blocks):
            requires_grad = i >= (total - finetune_blocks)
            for p in block.parameters():
                p.requires_grad = requires_grad
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True
        if hasattr(self.backbone, "pos_embed") and self.backbone.pos_embed is not None:
            self.backbone.pos_embed.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        x = self.backbone.forward_features(x)
        x = x[:, 1:, :]
        x = self.proj(x)
        return x
