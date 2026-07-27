"""
Transformer Block for AethyxLM.

Architecture (Pre-LayerNorm):

Input
   │
   ▼
LayerNorm
   │
   ▼
Multi-Head Self Attention
   │
   ▼
Residual Add
   │
   ▼
LayerNorm
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

from model.attention import MultiHeadSelfAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNorm


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
    ):
        super().__init__()

        # First LayerNorm
        self.norm1 = LayerNorm(embed_dim=embed_dim, eps=layer_norm_eps)

        # Self-Attention
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            context_length=context_length,
            use_bias=use_bias,
        )

        # Second LayerNorm
        self.norm2 = LayerNorm(embed_dim=embed_dim, eps=layer_norm_eps)

        # Feed Forward Network
        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=ffn_dim,
            dropout=dropout,
            use_bias=use_bias,
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