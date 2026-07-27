"""
Custom Layer Normalization implementation for AethyxLM.

Normalizes each token across its embedding dimension.
"""

import torch
import torch.nn as nn

from model.config import (
    EMBED_DIM,
    LAYER_NORM_EPS,
)


class LayerNorm(nn.Module):
    """
    Custom implementation of Layer Normalization.

    Input Shape:
        (batch_size, sequence_length, embed_dim)

    Output Shape:
        (batch_size, sequence_length, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = None,
        eps: float = None,
    ):
        super().__init__()

        if embed_dim is None:
            embed_dim = EMBED_DIM
        if eps is None:
            eps = LAYER_NORM_EPS
        super().__init__()

        self.embed_dim = embed_dim
        self.eps = eps

        # Learnable scale parameter (γ)
        self.gamma = nn.Parameter(
            torch.ones(embed_dim)
        )

        # Learnable bias parameter (β)
        self.beta = nn.Parameter(
            torch.zeros(embed_dim)
        )

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

        # Mean across embedding dimension
        mean = x.mean(
            dim=-1,
            keepdim=True,
        )

        # Variance across embedding dimension
        variance = x.var(
            dim=-1,
            unbiased=False,
            keepdim=True,
        )

        # Normalize
        x_hat = (x - mean) / torch.sqrt(
            variance + self.eps
        )

        # Scale and shift
        output = self.gamma * x_hat + self.beta

        return output