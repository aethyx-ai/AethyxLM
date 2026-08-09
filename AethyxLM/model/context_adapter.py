"""Opt-in latent interface for compiled or structured context.

This module deliberately does not prescribe whether compiler features originate
from text, graphs, visual pages, or another representation. It provides a fixed
latent budget and a clean cross-attention boundary for controlled experiments.
"""

from typing import Optional

import torch
import torch.nn as nn

from model.modules.rmsnorm import RMSNorm


class LatentContextAdapter(nn.Module):
    """Resample variable-length compiler features into fixed context latents."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_latents: int = 64,
        num_heads: int = 8,
        num_types: int = 16,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_latents <= 0:
            raise ValueError("num_latents must be positive")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by context num_heads")
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.input_projection = nn.Linear(input_dim, embed_dim, bias=False)
        self.type_embedding = nn.Embedding(num_types, embed_dim)
        self.latents = nn.Parameter(torch.empty(num_latents, embed_dim))
        nn.init.normal_(self.latents, std=0.02)

        self.cross_norms = nn.ModuleList(RMSNorm(embed_dim) for _ in range(depth))
        self.cross_attentions = nn.ModuleList(
            nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True, bias=False
            )
            for _ in range(depth)
        )
        self.ffn_norms = nn.ModuleList(RMSNorm(embed_dim) for _ in range(depth))
        self.ffns = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(4 * embed_dim, embed_dim, bias=False),
            )
            for _ in range(depth)
        )
        self.output_norm = RMSNorm(embed_dim)

    def forward(
        self,
        features: torch.Tensor,
        type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.size(-1) != self.input_dim:
            raise ValueError(
                f"context features must have shape (batch, items, {self.input_dim})"
            )
        context = self.input_projection(features)
        if type_ids is not None:
            if type_ids.shape != features.shape[:2]:
                raise ValueError("context type_ids must match the first two feature dimensions")
            context = context + self.type_embedding(type_ids)
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != features.shape[:2]:
                raise ValueError("context attention_mask has an invalid shape")
            key_padding_mask = ~attention_mask.bool()
            if key_padding_mask.all(dim=1).any():
                raise ValueError("every context batch row must contain at least one valid item")

        latents = self.latents.unsqueeze(0).expand(features.size(0), -1, -1)
        for norm, attention, ffn_norm, ffn in zip(
            self.cross_norms, self.cross_attentions, self.ffn_norms, self.ffns
        ):
            normalized = norm(latents)
            attended, _ = attention(
                normalized,
                context,
                context,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            latents = latents + attended
            latents = latents + ffn(ffn_norm(latents))
        return self.output_norm(latents)


class ContextCrossAttention(nn.Module):
    """Inject compiled-context latents into selected decoder layers."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True, bias=False
        )
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, hidden_states: torch.Tensor, context_latents: torch.Tensor):
        update, _ = self.attention(
            self.norm(hidden_states), context_latents, context_latents, need_weights=False
        )
        return hidden_states + torch.sigmoid(self.gate) * update
