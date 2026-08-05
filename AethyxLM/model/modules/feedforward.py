"""
Feed Forward Network (MLP) for AethyxLM.

This module implements both GELU and SwiGLU variants.

Reference:
    "GLU Variants Improve Transformer" - Shazeer (2020)
    https://arxiv.org/abs/2002.05202

SwiGLU(x) = SiLU(x) * (x @ W_gate) = sigmoid(x) * x * (x @ W_gate)

In practice, we split the input into two halves:
- x = [x1, x2] where x1, x2 each have dimension hidden_dim/2
- SwiGLU(x) = SiLU(x1) * x2

For fair comparison with GELU, we use 2/3 expansion:
- GELU: hidden_dim = 4 * embed_dim
- SwiGLU: hidden_dim = 8/3 * embed_dim (so that parameter count is similar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import (
    EMBED_DIM,
    FFN_DIM,
    DROPOUT,
    USE_BIAS,
    FFN_TYPE,
    SWIGLU_HIDDEN_MULT,
)

from model.layers import Linear


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network with GELU or SwiGLU activation.

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
        ffn_type: str = None,
    ):
        super().__init__()

        if embed_dim is None:
            embed_dim = EMBED_DIM
        if dropout is None:
            dropout = DROPOUT
        if use_bias is None:
            use_bias = USE_BIAS
        if ffn_type is None:
            ffn_type = FFN_TYPE

        self.ffn_type = ffn_type

        if ffn_type == "swiglu":
            # SwiGLU uses 2/3 expansion to match parameter count of GELU with 4x
            # SwiGLU splits hidden_dim into two halves, so we need 2 * (8/3) * embed_dim / 2 = 8/3 * embed_dim
            # But since we split into two, each half is 4/3 * embed_dim
            # So total hidden = 2 * (4/3 * embed_dim) = 8/3 * embed_dim ≈ 2.67 * embed_dim
            if hidden_dim is None:
                hidden_dim = int(embed_dim * SWIGLU_HIDDEN_MULT * 2 / 3)
            self.hidden_dim = hidden_dim
        else:
            if hidden_dim is None:
                hidden_dim = FFN_DIM
            self.hidden_dim = hidden_dim

        self.ffn_type = ffn_type

        if ffn_type == "gelu":
            # GELU: Linear -> GELU -> Dropout -> Linear
            self.fc1 = Linear(
                embed_dim,
                self.hidden_dim,
                bias=use_bias,
            )
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = Linear(
                self.hidden_dim,
                embed_dim,
                bias=use_bias,
            )

        elif ffn_type == "swiglu":
            # SwiGLU: Split -> SiLU * Gate -> Linear
            # We need 2 * hidden_dim for the split (gate and value)
            self.fc1 = Linear(
                embed_dim,
                2 * self.hidden_dim,
                bias=use_bias,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc2 = Linear(
                self.hidden_dim,
                embed_dim,
                bias=use_bias,
            )

        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}. Supported: 'gelu', 'swiglu'")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            Output tensor of shape (B, T, D)
        """
        if self.ffn_type == "gelu":
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = self.fc2(x)

        elif self.ffn_type == "swiglu":
            # Project to 2 * hidden_dim
            x = self.fc1(x)

            # Split into gate and value
            x_gate, x_val = x.chunk(2, dim=-1)

            # SwiGLU: SiLU(gate) * value
            x = F.silu(x_gate) * x_val

            x = self.dropout(x)
            x = self.fc2(x)

        return x


def build_feedforward(
    embed_dim: int,
    hidden_dim: int = None,
    dropout: float = 0.1,
    use_bias: bool = True,
    ffn_type: str = "gelu",
) -> nn.Module:
    """
    Factory function to build feedforward network.

    Args:
        embed_dim: Embedding dimension.
        hidden_dim: Hidden dimension (auto-computed if None).
        dropout: Dropout rate.
        use_bias: Whether to use bias in linear layers.
        ffn_type: "gelu" or "swiglu".

    Returns:
        FeedForward module.
    """
    return FeedForward(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_bias=use_bias,
        ffn_type=ffn_type,
    )