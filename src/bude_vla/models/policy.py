"""Full BUD-E policy: vision + text + proprio + soft prompts + backbone + action head.

The contract (data shapes) is defined here. Every component on every dimension
agrees via this class.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from bude_vla.models.action_head import ContextActionHead, DirectActionHead, FlowMatchingActionHead, GripperTriggerHead
from bude_vla.models.backbone import PolicyTransformer
from bude_vla.models.proprio import ProprioProjector
from bude_vla.models.soft_prompts import SoftPrompts
from bude_vla.models.text_encoder import MiniLMTextEncoder, TinyTextEncoder
from bude_vla.models.vision import DINOv2Tower, ViTSmall


@dataclass
class BUDEConfig:
    use_dinov2: bool = False
    use_minilm: bool = False
    dinov2_finetune_blocks: int = 4
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
    flow_n_steps: int = 50
    use_bc_head: bool = False
    use_visual_action_cond: bool = False
    use_context_action_head: bool = False
    use_perception: bool = False
    use_perception_action_cond: bool = False
    perception_dim: int = 3
    use_gripper_trigger_head: bool = False
    gripper_trigger_threshold: float = 0.5
    gripper_trigger_close_value: float = -1.0
    action_space: str = "joint_abs"  # joint_abs or ee_delta
    ee_delta_scale: float = 0.05


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
        self.perception_proj = (
            nn.Sequential(
                nn.LayerNorm(cfg.perception_dim),
                nn.Linear(cfg.perception_dim, cfg.d),
                nn.SiLU(),
                nn.Linear(cfg.d, cfg.d),
            )
            if cfg.use_perception else None
        )
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
        self.bc_action_head = (
            DirectActionHead(
                action_dim=cfg.action_dim,
                chunk_size=cfg.chunk_size,
                d=cfg.d,
                hidden_dim=cfg.action_head_hidden_dim,
            )
            if cfg.use_bc_head else None
        )
        self.action_cond_proj = (
            nn.Sequential(
                nn.LayerNorm(cfg.d * 3),
                nn.Linear(cfg.d * 3, cfg.d),
                nn.SiLU(),
                nn.Linear(cfg.d, cfg.d),
            )
            if cfg.use_visual_action_cond else None
        )
        context_cond_dim = cfg.d * 2 if (cfg.use_perception_action_cond and cfg.use_perception) else 0
        self.context_action_head = (
            ContextActionHead(
                action_dim=cfg.action_dim,
                chunk_size=cfg.chunk_size,
                d=cfg.d,
                hidden_dim=cfg.action_head_hidden_dim,
                depth=2,
                heads=cfg.backbone_heads,
                cond_dim=context_cond_dim,
            )
            if cfg.use_context_action_head else None
        )
        self.gripper_trigger_cond_dim = context_cond_dim if context_cond_dim > 0 else cfg.d
        self.gripper_trigger_head = (
            GripperTriggerHead(
                chunk_size=cfg.chunk_size,
                d=self.gripper_trigger_cond_dim,
                hidden_dim=cfg.action_head_hidden_dim,
            )
            if cfg.use_gripper_trigger_head else None
        )

        self.chunk_size = cfg.chunk_size
        self.action_dim = cfg.action_dim
        self.d = cfg.d

    def _patch_start_index(self) -> int:
        return self.cfg.n_prompts + 1 + int(self.perception_proj is not None)

    def _perception_input(self, batch: dict, ref: torch.Tensor) -> torch.Tensor:
        perception = batch.get("perception")
        if perception is None:
            return torch.zeros(
                ref.shape[0], self.cfg.perception_dim,
                dtype=ref.dtype, device=ref.device,
            )
        return perception.to(device=ref.device, dtype=ref.dtype)

    def _perception_embedding(self, batch: dict, ref: torch.Tensor) -> torch.Tensor | None:
        if self.perception_proj is None:
            return None
        return self.perception_proj(self._perception_input(batch, ref))

    def _context_action_cond(self, batch: dict, tokens: torch.Tensor) -> torch.Tensor | None:
        if not self.cfg.use_perception_action_cond or self.perception_proj is None:
            return None
        state_hidden = tokens[:, self.cfg.n_prompts, :]
        perception_emb = self._perception_embedding(batch, tokens)
        return torch.cat([state_hidden, perception_emb], dim=-1)


    def _gripper_trigger_cond(self, batch: dict, tokens: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_perception_action_cond and self.perception_proj is not None:
            return self._context_action_cond(batch, tokens)
        return tokens[:, self.cfg.n_prompts, :]

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

        token_parts = [prompts, state_token]
        if self.perception_proj is not None:
            perception_emb = self._perception_embedding(batch, images)
            token_parts.append(perception_emb.unsqueeze(1))
        token_parts.extend([patch_tokens, text_tokens])

        tokens = torch.cat(token_parts, dim=1)
        tokens = self.backbone(tokens)
        return tokens

    def forward(self, batch: dict) -> dict:
        """Training forward: compute predicted velocity field for action denoising."""
        actions = batch["actions"]      # (B, T, A)
        tau = batch["tau"]              # (B,)
        noise = batch["noise"]          # (B, T, A)

        tokens = self.encode(batch)
        # Pull the state token out (always at position n_prompts).
        # Optionally add pooled visual patch tokens so action decoding cannot
        # collapse to a proprio/phase-only policy.
        n_p = self.cfg.n_prompts
        state_hidden = tokens[:, n_p, :]                      # (B, d)
        if self.action_cond_proj is not None:
            n_patch = self.vision.num_patches if hasattr(self.vision, "num_patches") else self.vision.patch_embed.num_patches
            patch_start = self._patch_start_index()
            patches = tokens[:, patch_start:patch_start + n_patch, :]
            visual_mean = patches.mean(dim=1)
            visual_max = patches.max(dim=1).values
            state_hidden = self.action_cond_proj(torch.cat([state_hidden, visual_mean, visual_max], dim=-1))

        # x_t at training time: (1-tau) * noise + tau * actions
        # and the velocity target is actions - noise
        tau_b = tau.view(-1, 1, 1)
        x_t = (1.0 - tau_b) * noise + tau_b * actions
        v_pred = self.action_head(x_t, tau, state_hidden)
        out = {"velocity": v_pred, "tokens": tokens}
        trigger_cond = None
        if self.context_action_head is not None:
            trigger_cond = self._context_action_cond(batch, tokens)
            out["bc_actions"] = self.context_action_head(tokens, trigger_cond)
        elif self.bc_action_head is not None:
            trigger_cond = state_hidden
            out["bc_actions"] = self.bc_action_head(state_hidden)
        if self.gripper_trigger_head is not None:
            out["gripper_close_logits"] = self.gripper_trigger_head(
                self._gripper_trigger_cond(batch, tokens))
        return out

    @torch.no_grad()
    def sample(self, batch: dict) -> torch.Tensor:
        """Inference: predict the action chunk via flow matching."""
        tokens = self.encode(batch)
        if self.context_action_head is not None:
            actions = self.context_action_head(tokens, self._context_action_cond(batch, tokens))
            if self.gripper_trigger_head is not None:
                logits = self.gripper_trigger_head(self._gripper_trigger_cond(batch, tokens))
                close = torch.sigmoid(logits) >= self.cfg.gripper_trigger_threshold
                actions = actions.clone()
                actions[..., -1] = torch.where(
                    close,
                    torch.full_like(actions[..., -1], self.cfg.gripper_trigger_close_value),
                    actions[..., -1],
                )
            return actions
        n_p = self.cfg.n_prompts
        state_hidden = tokens[:, n_p, :]
        if self.action_cond_proj is not None:
            n_patch = self.vision.num_patches if hasattr(self.vision, "num_patches") else self.vision.patch_embed.num_patches
            patch_start = self._patch_start_index()
            patches = tokens[:, patch_start:patch_start + n_patch, :]
            visual_mean = patches.mean(dim=1)
            visual_max = patches.max(dim=1).values
            state_hidden = self.action_cond_proj(torch.cat([state_hidden, visual_mean, visual_max], dim=-1))
        if self.bc_action_head is not None:
            actions = self.bc_action_head(state_hidden)
        else:
            actions = self.action_head.sample(state_hidden)
        if self.gripper_trigger_head is not None:
            logits = self.gripper_trigger_head(self._gripper_trigger_cond(batch, tokens))
            close = torch.sigmoid(logits) >= self.cfg.gripper_trigger_threshold
            actions = actions.clone()
            actions[..., -1] = torch.where(
                close,
                torch.full_like(actions[..., -1], self.cfg.gripper_trigger_close_value),
                actions[..., -1],
            )
        return actions

    @torch.no_grad()
    def sample_flow(self, batch: dict) -> torch.Tensor:
        """Inference through the stochastic flow head, even when BC is enabled."""
        tokens = self.encode(batch)
        n_p = self.cfg.n_prompts
        state_hidden = tokens[:, n_p, :]
        if self.action_cond_proj is not None:
            n_patch = self.vision.num_patches if hasattr(self.vision, "num_patches") else self.vision.patch_embed.num_patches
            patch_start = self._patch_start_index()
            patches = tokens[:, patch_start:patch_start + n_patch, :]
            visual_mean = patches.mean(dim=1)
            visual_max = patches.max(dim=1).values
            state_hidden = self.action_cond_proj(torch.cat([state_hidden, visual_mean, visual_max], dim=-1))
        return self.action_head.sample(state_hidden)

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
        if self.bc_action_head is not None:
            out["bc_action_head"] = parts(self.bc_action_head)
        if self.action_cond_proj is not None:
            out["action_cond_proj"] = parts(self.action_cond_proj)
        if self.context_action_head is not None:
            out["context_action_head"] = parts(self.context_action_head)
        if self.perception_proj is not None:
            out["perception_proj"] = parts(self.perception_proj)
        if self.gripper_trigger_head is not None:
            out["gripper_trigger_head"] = parts(self.gripper_trigger_head)
        out["total"] = sum(out.values())
        return out
