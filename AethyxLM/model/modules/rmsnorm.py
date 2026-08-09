"""
RMSNorm (Root Mean Square Layer Normalization) for AethyxLM.

Reference:
    "Root Mean Square Layer Normalization" - Zhang & Sennrich (2019)
    https://arxiv.org/abs/1910.07467

RMSNorm normalizes inputs by their root mean square instead of
mean and variance like LayerNorm. It is more computationally efficient
and has been shown to work well in large language models.

Unlike LayerNorm, RMSNorm does not re-center the inputs (no mean subtraction),
only re-scales by the RMS of the activations.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Args:
        embed_dim: Dimension of the input features.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        embed_dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., embed_dim).

        Returns:
            Normalized tensor of same shape.
        """
        # Compute RMS along the last dimension
        # x^2 -> mean -> sqrt -> 1/rms
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Scale by weight
        return x * rms * self.weight

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


def build_normalization(
    embed_dim: int,
    normalization: str = "layernorm",
    eps: float = 1e-5,
) -> nn.Module:
    """
    Factory function to build normalization layer.

    Args:
        embed_dim: Embedding dimension.
        normalization: Type of normalization ("layernorm" | "rmsnorm").
        eps: Epsilon for numerical stability.

    Returns:
        Normalization module.

    Raises:
        ValueError: If normalization type is unknown.
    """
    if normalization == "layernorm":
        from model.layer_norm import LayerNorm
        return LayerNorm(embed_dim=embed_dim, eps=eps)
    elif normalization == "rmsnorm":
        return RMSNorm(embed_dim=embed_dim, eps=eps)
    else:
        raise ValueError(
            f"Unknown normalization type: {normalization}. "
            f"Supported: 'layernorm', 'rmsnorm'."
        )
