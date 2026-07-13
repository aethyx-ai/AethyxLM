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

from .attention import MultiHeadSelfAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    """
    GPT-style Transformer Block.

    Input:
        (batch_size, sequence_length, embed_dim)

    Output:
        (batch_size, sequence_length, embed_dim)
    """

    def __init__(self):
        super().__init__()

        # First LayerNorm
        self.norm1 = LayerNorm()

        # Self-Attention
        self.attention = MultiHeadSelfAttention()

        # Second LayerNorm
        self.norm2 = LayerNorm()

        # Feed Forward Network
        self.feed_forward = FeedForward()

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