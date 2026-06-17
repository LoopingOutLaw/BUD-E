"""Text encoders for BUD-E.

TinyTextEncoder  - from-scratch 4-layer BPE transformer (original, backward compat).
MiniLMTextEncoder - frozen pretrained all-MiniLM-L6-v2 with trainable projection.

This module exposes:
- `SimpleTokenizer`: a tiny integer-encoding tokenizer built on HuggingFace
  `tokenizers` (BPE). Used to train a vocab from instruction strings.
- `TinyTextEncoder`: nn.Module — token embeddings + sinusoidal pos embed +
  N-layer Transformer encoder + LayerNorm.
- `MiniLMTextEncoder`: nn.Module — frozen sentence-transformers MiniLM model
  with a trainable linear projection to the backbone dimension.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
RESERVED_SPECIALS = 4  # PAD, BOS, EOS, UNK


class SimpleTokenizer:
    """Small BPE tokenizer built on `tokenizers` library.

    train(corpus): builds a vocab from a list of strings.
    encode(text): returns list[int] with BOS + ids + EOS, padded to max_len.
    decode(ids):  reverse, strip specials.
    save(path) / load(path): JSON persistence.
    """

    def __init__(self, vocab_size: int = 512, max_len: int = 64):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self._tok: Tokenizer | None = None

    def train(self, corpus: Iterable[str]) -> None:
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
            initial_alphabet=[],
            show_progress=False,
        )
        tok.train_from_iterator(list(corpus), trainer)
        # Manually add BOS/EOS so they're always at start/end of sequences;
        # padding applied after to a fixed max_len so the encoder sees them.
        from tokenizers import pre_tokenizers
        from tokenizers.processors import TemplateProcessing
        tok.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[("<bos>", BOS_ID), ("<eos>", EOS_ID)],
        )
        tok.enable_padding(pad_id=PAD_ID, length=self.max_len)
        tok.enable_truncation(max_length=self.max_len)
        self._tok = tok

    def encode(self, text: str) -> list[int]:
        assert self._tok is not None, "Tokenizer not trained yet"
        return self._tok.encode(text).ids

    def batch_encode(self, texts: list[str]) -> torch.Tensor:
        assert self._tok is not None, "Tokenizer not trained yet"
        ids = [self._tok.encode(t).ids[: self.max_len] for t in texts]
        # pad to max_len
        out = torch.full((len(ids), self.max_len), PAD_ID, dtype=torch.long)
        for i, row in enumerate(ids):
            out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        return out

    def decode(self, ids: list[int]) -> str:
        assert self._tok is not None, "Tokenizer not trained yet"
        return self._tok.decode(ids, skip_special_tokens=True)

    def save(self, path: str | Path) -> None:
        assert self._tok is not None, "Tokenizer not trained yet"
        self._tok.save(str(path))

    def load(self, path: str | Path) -> None:
        self._tok = Tokenizer.from_file(str(path))


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional embeddings (DETR-style)."""

    def __init__(self, max_len: int, d: int):
        super().__init__()
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TinyTextEncoder(nn.Module):
    """Token + sin pos + N-layer Transformer encoder + LayerNorm.

    Input: (B, T_text) int64 token ids with PAD=0 at padding positions.
    Output: (B, T_text, d) hidden states (PAD positions are unmasked but don't
    carry semantic content, so the action head will ignore them via ctx tokens).
    """

    def __init__(self, vocab_size: int = 512, max_len: int = 64, d: int = 256,
                 depth: int = 4, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d, padding_idx=PAD_ID)
        self.pos_embed = SinusoidalPositionalEmbedding(max_len, d)
        self.pad_id = PAD_ID
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d)
        self.d = d

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(ids)
        x = self.pos_embed(x)
        # key_padding_mask: True at PAD positions
        kpm = (ids == self.pad_id)
        x = self.transformer(x, src_key_padding_mask=kpm)
        x = self.norm(x)
        return x


class MiniLMTextEncoder(nn.Module):
    """Frozen pretrained MiniLM (sentence-transformers/all-MiniLM-L6-v2) +
    trainable linear projection to the backbone dim.

    The MiniLM backbone produces per-token hidden states of shape (B, T, 384).
    We project to `d` so the output is (B, T, d) — same contract as
    TinyTextEncoder, so the policy doesn't need to know which encoder is active.

    All MiniLM parameters are frozen (`requires_grad=False`) so we only train
    the projection. This gives real semantic understanding of instructions
    ("pick", "push", "reach") learned from natural language pretraining,
    while keeping trainable param count low.

    Input: list[str] of instruction strings.
    Output: (B, max_seq_len, d)
    """

    def __init__(self, d: int = 256, max_len: int = 64,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer

        # Load model + tokenizer. SentenceTransformer wraps the BertModel.
        self._st = SentenceTransformer(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self._st._first_module().auto_model
        self.d = d
        self.max_len = max_len
        self.embed_dim = self.model.config.hidden_size

        for p in self.model.parameters():
            p.requires_grad = False
        if hasattr(self._st, "_modules"):
            for module in self._st._modules.values():
                for p in module.parameters():
                    p.requires_grad = False

        self.proj = nn.Linear(self.embed_dim, d)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep MiniLM frozen regardless of mode
        self.model.eval()
        return self

    def forward(self, texts) -> torch.Tensor:
        """texts: list[str] of instruction strings. Returns (B, T, d)."""
        device = next(self.proj.parameters()).device
        enc = self._tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = self.model(**enc).last_hidden_state
        B, T, _ = hidden.shape
        x = self.proj(hidden.view(B * T, -1)).view(B, T, self.d)
        return x
