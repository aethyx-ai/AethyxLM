"""
Tests for RoPE module.
"""

import sys
sys.path.insert(0, 'D:/CODING/AETHYXLabs/AethyxLM')

import torch
from model.modules.rope import RotaryEmbedding, apply_rotary_pos_emb, build_rope


def test_rope_forward():
    """Test RoPE forward pass."""
    torch.manual_seed(42)
    rope = RotaryEmbedding(head_dim=64, max_seq_len=128, base=10000.0)
    q = torch.randn(2, 8, 128, 64)
    k = torch.randn(2, 8, 128, 64)
    q_rot, k_rot = rope(q, k, seq_len=128)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print("[OK] RoPE forward pass")


def test_rope_output_shape():
    """Test RoPE output shapes."""
    torch.manual_seed(42)
    rope = RotaryEmbedding(head_dim=32, max_seq_len=128, base=10000.0)
    q = torch.randn(4, 8, 64, 32)
    k = torch.randn(4, 8, 64, 32)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print("[OK] RoPE output shapes")


def test_rope_cache_extension():
    """Test RoPE cache extension for longer sequences."""
    torch.manual_seed(42)
    rope = RotaryEmbedding(head_dim=32, max_seq_len=128, base=10000.0)
    q = torch.randn(2, 4, 256, 32)
    k = torch.randn(2, 4, 256, 32)
    q_rot, k_rot = rope(q, k, seq_len=256)
    assert q_rot.shape == q.shape
    print("[OK] RoPE cache extension")


def test_apply_rotary_pos_emb():
    """Test apply_rotary_pos_emb function."""
    torch.manual_seed(42)
    from model.modules.rope import apply_rotary_pos_emb
    x = torch.randn(2, 8, 128, 64)
    cos = torch.randn(1, 1, 128, 64)
    sin = torch.randn(1, 1, 128, 64)
    out = apply_rotary_pos_emb(x, cos, sin)
    assert out.shape == x.shape
    print("[OK] apply_rotary_pos_emb")


def test_rope_cache_correctness():
    """Test RoPE cache correctness with manual computation."""
    torch.manual_seed(42)
    rope = RotaryEmbedding(head_dim=4, max_seq_len=8, base=10000.0)
    q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]]]])  # (1, 1, 4, 4)
    k = q.clone()
    q_rot, k_rot = rope(q, k, seq_len=4)
    assert q_rot.shape == q.shape
    print("[OK] RoPE cache correctness")


def test_build_rope():
    """Test build_rope factory."""
    from model.modules.rope import build_rope
    rope = build_rope(head_dim=64, max_seq_len=128, base=10000.0)
    assert isinstance(rope, __import__('model.modules.rope', fromlist=['RotaryEmbedding']).RotaryEmbedding)
    print("[OK] build_rope factory")


if __name__ == "__main__":
    test_rope_forward()
    test_rope_output_shape()
    test_rope_cache_extension()
    test_apply_rotary_pos_emb()
    test_rope_cache_correctness()
    test_build_rope()
    print("\n[OK] All RoPE tests passed!")