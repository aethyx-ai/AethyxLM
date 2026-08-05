"""
Transformer Block for AethyxLM.

Architecture (Pre-LayerNorm):

Input
   │
   ▼
Normalization
   │
   ▼
Multi-Head Self Attention
   │
   ▼
Residual Add
   │
   ▼
Normalization
   │
   ▼
Feed Forward Network
   │
   ▼
Residual Add
   │
   ▼
Output
"""

import torch
import torch.nn as nn

from model.config import (
    EMBED_DIM,
    NUM_HEADS,
    FFN_DIM,
    DROPOUT,
    CONTEXT_LENGTH,
    USE_BIAS,
    LAYER_NORM_EPS,
    NORMALIZATION,
    FFN_TYPE,
)

from model.attention import MultiHeadSelfAttention
from model.modules.feedforward import FeedForward
from model.modules.rmsnorm import build_normalization


class TransformerBlock(nn.Module):
    """
    GPT-style Transformer Block.

    Input:
        (batch_size, sequence_length, embed_dim)

    Output:
        (batch_size, sequence_length, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = None,
        num_heads: int = None,
        ffn_dim: int = None,
        dropout: float = None,
        context_length: int = None,
        use_bias: bool = None,
        layer_norm_eps: float = None,
        normalization: str = None,
        ffn_type: str = None,
    ):
        super().__init__()

        if embed_dim is None:
            embed_dim = EMBED_DIM
        if num_heads is None:
            num_heads = NUM_HEADS
        if ffn_dim is None:
            ffn_dim = FFN_DIM
        if dropout is None:
            dropout = DROPOUT
        if context_length is None:
            context_length = CONTEXT_LENGTH
        if use_bias is None:
            use_bias = USE_BIAS
        if layer_norm_eps is None:
            layer_norm_eps = LAYER_NORM_EPS
        if normalization is None:
            normalization = NORMALIZATION
        if ffn_type is None:
            ffn_type = FFN_TYPE

        # First Normalization
        self.norm1 = build_normalization(
            embed_dim=embed_dim,
            normalization=normalization,
            eps=layer_norm_eps,
        )

        # Self-Attention
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            context_length=context_length,
            use_bias=use_bias,
        )

        # Second Normalization
        self.norm2 = build_normalization(
            embed_dim=embed_dim,
            normalization=normalization,
            eps=layer_norm_eps,
        )

        # Feed Forward Network
        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=ffn_dim,
            dropout=dropout,
            use_bias=use_bias,
            ffn_type=ffn_type,
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
            Tensor:
                Shape -> (B, T, D)
        """

        # --------------------------------------------
        # Attention Block
        # --------------------------------------------

        residual = x

        x = self.norm1(x)

        x = self.attention(x)

        x = x + residual

        # --------------------------------------------
        # Feed Forward Block
        # --------------------------------------------

        residual = x

        x = self.norm2(x)

        x = self.feed_forward(x)

        x = x + residual

        return x