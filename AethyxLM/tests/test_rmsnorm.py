"""
Tests for RMSNorm module.
"""

import sys
sys.path.insert(0, 'D:/CODING/AETHYXLabs/AethyxLM')

import torch
from model.modules.rmsnorm import RMSNorm, build_normalization


def test_rmsnorm_forward():
    """Test RMSNorm forward pass."""
    torch.manual_seed(42)
    norm = RMSNorm(embed_dim=64, eps=1e-5)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == x.shape
    print("[OK] RMSNorm forward pass")


def test_rmsnorm_normalization():
    """Test that RMSNorm normalizes correctly."""
    torch.manual_seed(42)
    norm = RMSNorm(embed_dim=64, eps=1e-5)
    x = torch.randn(100, 64)
    out = norm(x)
    # Check RMS is close to 1
    rms = torch.sqrt(out.pow(2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)
    print("[OK] RMSNorm normalization correct")


def test_rmsnorm_vs_layernorm():
    """Test RMSNorm vs LayerNorm behavior."""
    torch.manual_seed(42)
    from model.modules.rmsnorm import build_normalization
    import torch.nn as nn

    rms = build_normalization(64, "rmsnorm", 1e-5)
    ln = nn.LayerNorm(64, eps=1e-5)

    x = torch.randn(2, 10, 64)
    out_rms = rms(x)
    out_ln = ln(x)

    assert out_rms.shape == out_ln.shape
    print("[OK] RMSNorm vs LayerNorm shape compatibility")


def test_build_normalization():
    """Test build_normalization factory."""
    from model.modules.rmsnorm import build_normalization

    rms = build_normalization(64, "rmsnorm", 1e-5)
    assert isinstance(rms, __import__('model.modules.rmsnorm', fromlist=['RMSNorm']).RMSNorm)

    from model.layer_norm import LayerNorm
    ln = build_normalization(64, "layernorm", 1e-5)
    assert isinstance(ln, __import__('model.layer_norm', fromlist=['LayerNorm']).LayerNorm)

    try:
        build_normalization(64, "unknown", 1e-5)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    print("[OK] build_normalization factory")


if __name__ == "__main__":
    test_rmsnorm_forward()
    test_rmsnorm_normalization()
    test_rmsnorm_vs_layernorm()
    test_build_normalization()
    print("\n[OK] All RMSNorm tests passed!")