"""Efficient causal self-attention for AethyxLM.

Supports multi-head and grouped-query attention, PyTorch SDPA/Flash kernels,
optional fused QKV projections, RoPE, QK normalization, and inference KV cache.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import (
    CONTEXT_LENGTH,
    DROPOUT,
    EMBED_DIM,
    NUM_HEADS,
    POSITION_ENCODING,
    ROPE_BASE,
    ROPE_MAX_SEQ_LEN,
    USE_BIAS,
)
from model.layers import Linear
from model.modules.rope import RotaryEmbedding


KVCache = Tuple[torch.Tensor, torch.Tensor]


class MultiHeadSelfAttention(nn.Module):
    """Causal self-attention with optional grouped key/value heads."""

    def __init__(
        self,
        embed_dim: int = None,
        num_heads: int = None,
        num_kv_heads: int = None,
        dropout: float = None,
        context_length: int = None,
        use_bias: bool = None,
        position_encoding: str = None,
        rope_base: float = None,
        rope_max_seq_len: int = None,
        rope_scaling_factor: float = 1.0,
        fused_qkv: bool = False,
        use_sdpa: bool = True,
        qk_norm: bool = False,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        embed_dim = EMBED_DIM if embed_dim is None else embed_dim
        num_heads = NUM_HEADS if num_heads is None else num_heads
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        dropout = DROPOUT if dropout is None else dropout
        context_length = CONTEXT_LENGTH if context_length is None else context_length
        use_bias = USE_BIAS if use_bias is None else use_bias
        position_encoding = POSITION_ENCODING if position_encoding is None else position_encoding
        rope_base = ROPE_BASE if rope_base is None else rope_base
        rope_max_seq_len = ROPE_MAX_SEQ_LEN if rope_max_seq_len is None else rope_max_seq_len

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.dropout_rate = dropout
        self.position_encoding = position_encoding
        self.fused_qkv = fused_qkv
        self.use_sdpa = use_sdpa and hasattr(F, "scaled_dot_product_attention")
        self.qk_norm = qk_norm
        if sliding_window is not None and sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        self.sliding_window = sliding_window

        if fused_qkv:
            self.qkv_proj = Linear(
                embed_dim, embed_dim + 2 * self.kv_dim, bias=use_bias
            )
        else:
            # This layout preserves compatibility with existing checkpoints.
            self.q_proj = Linear(embed_dim, embed_dim, bias=use_bias)
            self.k_proj = Linear(embed_dim, self.kv_dim, bias=use_bias)
            self.v_proj = Linear(embed_dim, self.kv_dim, bias=use_bias)

        self.out_proj = Linear(embed_dim, embed_dim, bias=use_bias)
        self.resid_dropout = nn.Dropout(dropout)

        # Kept only for the portable/manual attention fallback.
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

        self.rope = (
            RotaryEmbedding(
                self.head_dim, rope_max_seq_len, rope_base, rope_scaling_factor
            )
            if position_encoding == "rope"
            else None
        )

    def _project(self, x: torch.Tensor):
        if self.fused_qkv:
            projected = self.qkv_proj(x)
            return torch.split(
                projected, [self.embed_dim, self.kv_dim, self.kv_dim], dim=-1
            )
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def _expand_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        """Expand grouped K/V heads without allocating when possible."""
        if self.num_kv_heads == self.num_heads:
            return tensor
        repeats = self.num_heads // self.num_kv_heads
        batch, kv_heads, seq_len, head_dim = tensor.shape
        return (
            tensor[:, :, None, :, :]
            .expand(batch, kv_heads, repeats, seq_len, head_dim)
            .reshape(batch, self.num_heads, seq_len, head_dim)
        )

    def _attention_mask(
        self,
        query_length: int,
        key_length: int,
        position_offset: int,
        key_start_position: int,
        device: torch.device,
    ) -> torch.Tensor:
        query_positions = torch.arange(
            position_offset, position_offset + query_length, device=device
        )
        key_positions = torch.arange(
            key_start_position, key_start_position + key_length, device=device
        )
        mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if self.sliding_window is not None:
            mask &= key_positions.unsqueeze(0) > (
                query_positions.unsqueeze(1) - self.sliding_window
            )
        return mask

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ):
        batch_size, query_length, _ = x.shape
        cached_length = 0 if kv_cache is None else kv_cache[0].size(2)
        position_offset = (
            0
            if kv_cache is None
            else int(kv_cache[2]) if len(kv_cache) > 2 else cached_length
        )

        q, k, v = self._project(x)
        q = q.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, query_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, query_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(
                q, k, seq_len=query_length, position_offset=position_offset
            )

        if self.qk_norm:
            # Unit RMS keeps dot products controlled while preserving SDPA scaling.
            q = F.normalize(q.float(), dim=-1).to(q.dtype) * self.head_dim**0.5
            k = F.normalize(k.float(), dim=-1).to(k.dtype) * self.head_dim**0.5

        if kv_cache is not None:
            cached_k, cached_v = kv_cache[:2]
            k = torch.cat((cached_k, k), dim=2)
            v = torch.cat((cached_v, v), dim=2)
        if (
            self.sliding_window is not None
            and query_length == 1
            and k.size(2) > self.sliding_window
        ):
            k = k[:, :, -self.sliding_window :]
            v = v[:, :, -self.sliding_window :]
        present = None
        if use_cache:
            present_k, present_v = k, v
            if self.sliding_window is not None:
                present_k = present_k[:, :, -self.sliding_window :]
                present_v = present_v[:, :, -self.sliding_window :]
            present = (present_k, present_v, position_offset + query_length)

        expanded_k = self._expand_kv(k)
        expanded_v = self._expand_kv(v)
        key_length = expanded_k.size(2)
        dropout_p = self.dropout_rate if self.training else 0.0

        if self.use_sdpa:
            if position_offset == 0 and self.sliding_window is None:
                output = F.scaled_dot_product_attention(
                    q, expanded_k, expanded_v, dropout_p=dropout_p, is_causal=True
                )
            elif query_length == 1:
                # During token-by-token decoding every retained cache key is in
                # the past (or is the current token), so no mask is necessary.
                output = F.scaled_dot_product_attention(
                    q,
                    expanded_k,
                    expanded_v,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
            else:
                key_start = position_offset - cached_length
                mask = self._attention_mask(
                    query_length, key_length, position_offset, key_start, x.device
                )
                output = F.scaled_dot_product_attention(
                    q,
                    expanded_k,
                    expanded_v,
                    attn_mask=mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
        else:
            scores = q @ expanded_k.transpose(-2, -1) / self.head_dim**0.5
            key_start = position_offset - cached_length
            mask = self._attention_mask(
                query_length, key_length, position_offset, key_start, x.device
            )
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
            weights = F.dropout(weights, dropout_p, training=self.training)
            output = weights @ expanded_v

        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_length, self.embed_dim
        )
        output = self.resid_dropout(self.out_proj(output))
        return (output, present) if use_cache else output
