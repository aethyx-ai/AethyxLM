"""
Rotary Positional Embeddings (RoPE) for AethyxLM.

Reference:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding" - Su et al. (2021)
    https://arxiv.org/abs/2104.09864

RoPE encodes absolute position information by rotating query and key vectors
in a way that the dot product between positions depends only on their
relative distance. This allows the model to naturally extrapolate to
sequence lengths longer than seen during training.

Key properties:
- No learned positional embeddings needed
- Relative position encoding baked into attention
- Extrapolates to longer sequences than training context
- Applied only to Q and K (not V)
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).

    Computes and caches sin/cos values for rotary position embeddings.
    Supports automatic cache extension for sequences longer than
    precomputed length.

    Args:
        head_dim: Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length for precomputed cache.
        base: Base frequency for RoPE (theta in paper).
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        # theta_i = base^(-2i/d) for i in [0, head_dim/2)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute sin/cos cache up to max_seq_len
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len: int) -> None:
        """Update sin/cos cache for the given sequence length."""
        device = self.inv_freq.device
        dtype = self.inv_freq.dtype

        # t = [0, 1, ..., seq_len - 1]
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # freqs = t * inv_freq  -> (seq_len, head_dim // 2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Embedding dimension is head_dim // 2, so we need 2 * (head_dim // 2) = head_dim
        # sin and cos for each frequency
        # Shape: (seq_len, head_dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: Query tensor of shape (B, H, T, Hd).
            k: Key tensor of shape (B, H, T, Hd).
            seq_len: Optional sequence length. If None, inferred from q.

        Returns:
            Tuple of (q_rotated, k_rotated) with RoPE applied.
        """
        if seq_len is None:
            seq_len = q.size(2)

        # Ensure cache is large enough
        if seq_len > self.cos_cached.size(0):
            self._update_cache(seq_len)

        # Get sin/cos for the sequence length
        # Shape: (seq_len, head_dim)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # Reshape for broadcasting: (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply RoPE to q and k
        q_rot = apply_rotary_pos_emb(q, cos, sin)
        k_rot = apply_rotary_pos_emb(k, cos, sin)

        return q_rot, k_rot


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary positional embeddings to input tensor.

    Args:
        x: Input tensor of shape (B, H, T, Hd).
        cos: Cosine cache of shape (1, 1, T, Hd).
        sin: Sine cache of shape (1, 1, T, Hd).

    Returns:
        Rotated tensor of same shape.
    """
    # Split into two halves for rotation
    # x = [x1, x2] where x1, x2 are the two halves of the head_dim
    x1, x2 = x.chunk(2, dim=-1)

    # Rotation: x' = x * cos + (-x2, x1) * sin
    # Rotated = x * cos + rotate_half(x) * sin
    x_rot = (x1 * cos[..., :x.size(-1)//2]) - (x2 * sin[..., :x.size(-1)//2])
    x_rot2 = (x1 * sin[..., :x.size(-1)//2]) + (x2 * cos[..., :x.size(-1)//2])

    return torch.cat((x_rot, x_rot2), dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dimensions.

    Args:
        x: Tensor of shape (..., 2*H).

    Returns:
        Tensor with halves swapped and negated appropriately.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rope(
    head_dim: int,
    max_seq_len: int = 8192,
    base: float = 10000.0,
) -> RotaryEmbedding:
    """
    Factory function to build RoPE module.

    Args:
        head_dim: Dimension of each attention head.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency.

    Returns:
        RotaryEmbedding module.
    """
    return RotaryEmbedding(
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        base=base,
    )