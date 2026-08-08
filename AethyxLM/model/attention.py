"""
Multi-Head Self-Attention for AethyxLM.

This module implements the core attention mechanism used by GPT-style
decoder-only transformers.

Pipeline:

Input
    │
    ▼
Q, K, V Projections
    │
    ▼
Split into Multiple Heads
    │
    ▼
RoPE (optional)
    │
    ▼
Scaled Dot Product Attention
    │
    ▼
Causal Mask
    │
    ▼
Softmax
    │
    ▼
Weighted Sum
    │
    ▼
Concatenate Heads
    │
    ▼
Output Projection
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import (
    EMBED_DIM,
    NUM_HEADS,
    HEAD_DIM,
    CONTEXT_LENGTH,
    DROPOUT,
    USE_BIAS,
    POSITION_ENCODING,
    ROPE_BASE,
    ROPE_MAX_SEQ_LEN,
)

from model.layers import Linear
from model.modules.rope import RotaryEmbedding


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    Input Shape:
        (batch_size, sequence_length, embed_dim)

    Output Shape:
        (batch_size, sequence_length, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = None,
        num_heads: int = None,
        dropout: float = None,
        context_length: int = None,
        use_bias: bool = None,
        position_encoding: str = None,
        rope_base: float = None,
        rope_max_seq_len: int = None,
    ):
        super().__init__()

        if embed_dim is None:
            embed_dim = EMBED_DIM
        if num_heads is None:
            num_heads = NUM_HEADS
        if dropout is None:
            dropout = DROPOUT
        if context_length is None:
            context_length = CONTEXT_LENGTH
        if use_bias is None:
            use_bias = USE_BIAS
        if position_encoding is None:
            position_encoding = POSITION_ENCODING
        if rope_base is None:
            rope_base = ROPE_BASE
        if rope_max_seq_len is None:
            rope_max_seq_len = ROPE_MAX_SEQ_LEN

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.position_encoding = position_encoding

        # --------------------------------------------------
        # Query, Key and Value projections
        # --------------------------------------------------

        self.q_proj = Linear(
            embed_dim,
            embed_dim,
            bias=use_bias,
        )

        self.k_proj = Linear(
            embed_dim,
            embed_dim,
            bias=use_bias,
        )

        self.v_proj = Linear(
            embed_dim,
            embed_dim,
            bias=use_bias,
        )

        # --------------------------------------------------
        # Output Projection
        # --------------------------------------------------

        self.out_proj = Linear(
            embed_dim,
            embed_dim,
            bias=use_bias,
        )

        self.dropout = nn.Dropout(dropout)

        # --------------------------------------------------
        # Causal Mask
        # --------------------------------------------------

        mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
            )
        )

        self.register_buffer(
            "causal_mask",
            mask,
        )

        # --------------------------------------------------
        # RoPE (if enabled)
        # --------------------------------------------------

        if position_encoding == "rope":
            self.rope = RotaryEmbedding(
                head_dim=self.head_dim,
                max_seq_len=rope_max_seq_len,
                base=rope_base,
            )
        else:
            self.rope = None

        # --------------------------------------------------
        # Causal Mask
        # --------------------------------------------------

        mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
            )
        )

        self.register_buffer(
            "causal_mask",
            mask,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:
                Shape -> (B, T, D)

        Returns:
            Tensor
                Shape -> (B, T, D)
        """

        batch_size, sequence_length, _ = x.shape

        # --------------------------------------------
        # Project into Q, K and V
        # --------------------------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # --------------------------------------------
        # Split into heads
        # --------------------------------------------

        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        # --------------------------------------------
        # Move head dimension forward
        #
        # (B, T, H, Hd)
        #
        # becomes
        #
        # (B, H, T, Hd)
        # --------------------------------------------

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # --------------------------------------------
        # Apply RoPE to Q and K (if enabled)
        # --------------------------------------------

        if self.rope is not None:
            q, k = self.rope(q, k, seq_len=sequence_length)

        # --------------------------------------------
        # Scaled Dot Product Attention
        # --------------------------------------------

        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # --------------------------------------------
        # Apply causal mask
        # --------------------------------------------

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        attention_scores = attention_scores.masked_fill(
            mask == 0,
            float("-inf"),
        )

        # --------------------------------------------
        # Convert scores to probabilities
        # --------------------------------------------

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights,
        )

        # --------------------------------------------
        # Compute weighted sum of values
        #
        # (B, H, T, T)
        #     @
        # (B, H, T, Hd)
        #
        # ->
        #
        # (B, H, T, Hd)
        # --------------------------------------------

        attention_output = attention_weights @ v

        # --------------------------------------------
        # Move the head dimension back
        #
        # (B, H, T, Hd)
        #
        # becomes
        #
        # (B, T, H, Hd)
        # --------------------------------------------

        attention_output = attention_output.transpose(1, 2)

        # --------------------------------------------
        # Concatenate all heads
        #
        # (B, T, H, Hd)
        #
        # ->
        #
        # (B, T, D)
        # --------------------------------------------

        attention_output = attention_output.contiguous().view(
            batch_size,
            sequence_length,
            self.embed_dim,
        )

        # --------------------------------------------
        # Final output projection
        # --------------------------------------------

        output = self.out_proj(attention_output)

        output = self.dropout(output)

        return output
