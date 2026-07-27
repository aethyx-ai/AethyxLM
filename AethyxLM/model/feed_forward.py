"""
Feed Forward Network (MLP) for AethyxLM.

This module is applied independently to every token after the
self-attention layer.
"""

import torch
import torch.nn as nn

from .config import (
    EMBED_DIM,
    FFN_DIM,
    DROPOUT,
    USE_BIAS,
)

from .layers import Linear


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.

    Input Shape:
        (batch_size, sequence_length, embed_dim)

    Output Shape:
        (batch_size, sequence_length, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = None,
        hidden_dim: int = None,
        dropout: float = None,
        use_bias: bool = None,
    ):
        super().__init__()

        if embed_dim is None:
            embed_dim = EMBED_DIM
        if hidden_dim is None:
            hidden_dim = FFN_DIM
        if dropout is None:
            dropout = DROPOUT
        if use_bias is None:
            use_bias = USE_BIAS

        self.fc1 = Linear(
            embed_dim,
            hidden_dim,
            bias=use_bias,
        )

        self.activation = nn.GELU()

        self.fc2 = Linear(
            hidden_dim,
            embed_dim,
            bias=use_bias,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Shape -> (B, T, D)

        Returns:
            Tensor:
                Shape -> (B, T, D)
        """

        x = self.fc1(x)

        x = self.activation(x)

        x = self.fc2(x)

        x = self.dropout(x)

        return x