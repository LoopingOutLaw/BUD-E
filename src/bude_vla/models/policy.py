"""Full BUD-E policy: vision + text + proprio + soft prompts + backbone + action head.

The contract (data shapes) is defined here. Every component on every dimension
agrees via this class.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from bude_vla.models.action_head import FlowMatchingActionHead
from bude_vla.models.backbone import PolicyTransformer
from bude_vla.models.proprio import ProprioProjector
from bude_vla.models.soft_prompts import SoftPrompts
from bude_vla.models.text_encoder import MiniLMTextEncoder, TinyTextEncoder
from bude_vla.models.vision import DINOv2Tower, ViTSmall


@dataclass
class BUDEConfig:
    use_dinov2: bool = False
    use_minilm: bool = False
    dinov2_finetune_blocks: int = 4  # last N DINOv2 blocks to fine-tune
    n_history_frames: int = 1  # 1 = single frame (Markovian)
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 6  # dual-cam; will multiply by n_history_frames
    vision_dim: int = 192
    vision_depth: int = 8
    vision_heads: int = 3
    text_vocab: int = 512
    text_max_len: int = 64
    text_depth: int = 4
    text_heads: int = 4
    state_dim: int = 6
    d: int = 256
    backbone_depth: int = 8
    backbone_heads: int = 8
    ffn_dim: int = 1024
    action_dim: int = 6
    chunk_size: int = 32
    n_domains: int = 6
    n_prompts: int = 32
    action_head_time_dim: int = 128
    action_head_hidden_dim: int = 512
    action_head_temporal_depth: int = 2  # P1: causal transformer layers
    action_head_temporal_heads: int = 4
    use_temporal_head: bool = True
    flow_n_steps: int = 10


class BUDEPolicy(nn.Module):
    """Vision-Language-Action policy.

    Forward signature (training):
        batch = {
            "images":    (B, 6, H, W),
            "text_ids":  (B, T_text),
            "proprio":   (B, state_dim),
            "domain_id": (B,) int64,
            "actions":   (B, chunk_size, action_dim),  # ground-truth actions
        }
        out = policy(batch)
        loss = ((out["velocity"] - velocity_target) ** 2).mean()

    Inference:
        actions = policy.sample({"images":..., "text_ids":..., "proprio":...,
                                 "domain_id":...})  # (B, chunk_size, action_dim)
    """

    def __init__(self, cfg: BUDEConfig | None = None):
        super().__init__()
        cfg = cfg or BUDEConfig()
        self.cfg = cfg

        # History stacking: channels are concatenated across time
        in_channels = cfg.in_channels * cfg.n_history_frames

        if cfg.use_dinov2:
            self.vision = DINOv2Tower(
                img_size=cfg.img_size,
                in_channels=in_channels,
                out_dim=cfg.d,
                finetune_blocks=cfg.dinov2_finetune_blocks,
            )
        else:
            self.vision = ViTSmall(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                in_channels=in_channels,
                dim=cfg.vision_dim,
                depth=cfg.vision_depth,
                heads=cfg.vision_heads,
                mlp_ratio=4.0,
                out_dim=cfg.d,
            )
        if cfg.use_minilm:
            self.text = MiniLMTextEncoder(
                d=cfg.d,
                max_len=cfg.text_max_len,
            )
        else:
            self.text = TinyTextEncoder(
                vocab_size=cfg.text_vocab,
                max_len=cfg.text_max_len,
                d=cfg.d,
                depth=cfg.text_depth,
                heads=cfg.text_heads,
            )
        self.proprio = ProprioProjector(state_dim=cfg.state_dim, out_dim=cfg.d)
        self.soft_prompts = SoftPrompts(n_domains=cfg.n_domains,
                                        n_prompts=cfg.n_prompts, d=cfg.d)
        self.backbone = PolicyTransformer(
            d=cfg.d, depth=cfg.backbone_depth,
            heads=cfg.backbone_heads, ffn_dim=cfg.ffn_dim,
        )
        self.action_head = FlowMatchingActionHead(
            action_dim=cfg.action_dim,
            chunk_size=cfg.chunk_size,
            d=cfg.d,
            time_dim=cfg.action_head_time_dim,
            hidden_dim=cfg.action_head_hidden_dim,
            n_steps=cfg.flow_n_steps,
            temporal_depth=cfg.action_head_temporal_depth if cfg.use_temporal_head else 0,
            temporal_heads=cfg.action_head_temporal_heads,
        )

        self.chunk_size = cfg.chunk_size
        self.action_dim = cfg.action_dim
        self.d = cfg.d

    def encode(self, batch: dict) -> torch.Tensor:
        """Build the input token sequence and run the backbone.

        Returns the full hidden-state sequence of shape (B, total_tokens, d).
        The state_token is at index `n_prompts` (the position after soft prompts).
        """
        images = batch["images"]
        proprio = batch["proprio"]
        domain_id = batch["domain_id"]
        if domain_id.dtype != torch.long:
            domain_id = domain_id.long()

        patch_tokens = self.vision(images)              # (B, N_patch, d)
        if self.cfg.use_minilm:
            text_tokens = self.text(batch["instruction"])
        else:
            text_tokens = self.text(batch["text_ids"])  # (B, T_text, d)
        state_token = self.proprio(proprio).unsqueeze(1)  # (B, 1, d)
        prompts = self.soft_prompts.gather(domain_id)    # (B, N_p, d)

        tokens = torch.cat([prompts, state_token, patch_tokens, text_tokens], dim=1)
        tokens = self.backbone(tokens)
        return tokens

    def forward(self, batch: dict) -> dict:
        """Training forward: compute predicted velocity field for action denoising."""
        actions = batch["actions"]      # (B, T, A)
        tau = batch["tau"]              # (B,)
        noise = batch["noise"]          # (B, T, A)

        tokens = self.encode(batch)
        # Pull the state token out (always at position n_prompts)
        n_p = self.cfg.n_prompts
        state_hidden = tokens[:, n_p, :]                      # (B, d)

        # x_t at training time: (1-tau) * noise + tau * actions
        # and the velocity target is actions - noise
        tau_b = tau.view(-1, 1, 1)
        x_t = (1.0 - tau_b) * noise + tau_b * actions
        v_pred = self.action_head(x_t, tau, state_hidden)
        return {"velocity": v_pred, "tokens": tokens}

    @torch.no_grad()
    def sample(self, batch: dict) -> torch.Tensor:
        """Inference: predict the action chunk via flow matching."""
        tokens = self.encode(batch)
        n_p = self.cfg.n_prompts
        state_hidden = tokens[:, n_p, :]
        actions = self.action_head.sample(state_hidden)
        return actions

    def n_params(self) -> dict[str, int]:
        def parts(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        out = {
            "vision": parts(self.vision),
            "text": parts(self.text),
            "proprio": parts(self.proprio),
            "soft_prompts": parts(self.soft_prompts),
            "backbone": parts(self.backbone),
            "action_head": parts(self.action_head),
        }
        out["total"] = sum(out.values())
        return out
