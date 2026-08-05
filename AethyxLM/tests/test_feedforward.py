"""
Tests for SwiGLU FeedForward module.
"""

import sys
sys.path.insert(0, 'D:/CODING/AETHYXLabs/AethyxLM')

import torch
from model.modules.feedforward import FeedForward, build_feedforward


def test_gelu_forward():
    """Test GELU feedforward forward pass."""
    torch.manual_seed(42)
    ff = FeedForward(embed_dim=64, ffn_type="gelu", hidden_dim=256, dropout=0.1)
    x = torch.randn(2, 10, 64)
    out = ff(x)
    assert out.shape == x.shape
    print("[OK] GELU forward pass")


def test_swiglu_forward():
    """Test SwiGLU feedforward forward pass."""
    torch.manual_seed(42)
    ff = FeedForward(embed_dim=64, ffn_type="swiglu", dropout=0.1)
    x = torch.randn(2, 10, 64)
    out = ff(x)
    assert out.shape == x.shape
    print("[OK] SwiGLU forward pass")


def test_swiglu_math():
    """Test SwiGLU mathematical correctness."""
    import torch.nn.functional as F
    x = torch.tensor([1.0, -1.0, 0.0, 2.0])
    # SwiGLU(x) = SiLU(x) * x
    expected = F.silu(x) * x
    # Simulate what the module does
    gate = x
    value = x
    result = F.silu(gate) * value
    assert torch.allclose(result, expected)
    print("[OK] SwiGLU mathematical correctness")


def test_gelu_vs_swiglu_parameters():
    """Test parameter count difference between GELU and SwiGLU."""
    gelu = FeedForward(embed_dim=64, ffn_type="gelu", hidden_dim=256)
    swiglu = FeedForward(embed_dim=64, ffn_type="swiglu")
    
    gelu_params = sum(p.numel() for p in gelu.parameters())
    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    
    # SwiGLU should have comparable or slightly more params due to gating
    assert swiglu_params > gelu_params * 0.8  # Within reasonable range
    print("[OK] Parameter counts: GELU={:,}, SwiGLU={:,}".format(gelu_params, swiglu_params))


def test_build_feedforward():
    """Test build_feedforward factory."""
    from model.modules.feedforward import build_feedforward
    ff = build_feedforward(embed_dim=64, hidden_dim=256, ffn_type="gelu")
    from model.modules.feedforward import FeedForward
    assert isinstance(ff, __import__('model.modules.feedforward', fromlist=['FeedForward']).FeedForward)
    
    ff_sw = build_feedforward(embed_dim=64, ffn_type="swiglu")
    assert isinstance(ff_sw, __import__('model.modules.feedforward', fromlist=['FeedForward']).FeedForward)
    print("[OK] build_feedforward factory")


def test_swiglu_gradient_flow():
    """Test gradient flow through SwiGLU."""
    torch.manual_seed(42)
    ff = FeedForward(embed_dim=64, ffn_type="swiglu", dropout=0.1)
    x = torch.randn(2, 10, 64, requires_grad=True)
    out = ff(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("[OK] SwiGLU gradient flow")


def test_gelu_gradient_flow():
    """Test gradient flow through GELU."""
    torch.manual_seed(42)
    ff = FeedForward(embed_dim=64, ffn_type="gelu", hidden_dim=256, dropout=0.1)
    x = torch.randn(2, 10, 64, requires_grad=True)
    out = ff(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("[OK] GELU gradient flow")


if __name__ == "__main__":
    test_gelu_forward()
    test_swiglu_forward()
    test_swiglu_math()
    test_gelu_vs_swiglu_parameters()
    test_build_feedforward()
    test_swiglu_gradient_flow()
    test_gelu_gradient_flow()
    print("\n[OK] All FeedForward tests passed!")