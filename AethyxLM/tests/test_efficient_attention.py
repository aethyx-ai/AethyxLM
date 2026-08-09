import torch

from model.attention import MultiHeadSelfAttention
from model.gpt import GPT


def tiny_config(**overrides):
    config = {
        "vocab_size": 97,
        "context_length": 32,
        "embed_dim": 32,
        "num_heads": 4,
        "num_kv_heads": 2,
        "num_layers": 2,
        "ffn_dim": 64,
        "dropout": 0.0,
        "use_bias": False,
        "normalization": "rmsnorm",
        "position_encoding": "rope",
        "ffn_type": "swiglu",
        "fused_qkv": True,
        "use_sdpa": True,
    }
    config.update(overrides)
    return config


def test_grouped_query_attention_shapes_and_cache_size():
    attention = MultiHeadSelfAttention(
        embed_dim=32,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.0,
        position_encoding="rope",
        fused_qkv=True,
    ).eval()
    output, cache = attention(torch.randn(2, 5, 32), use_cache=True)
    assert output.shape == (2, 5, 32)
    assert cache[0].shape == (2, 2, 5, 8)
    assert cache[1].shape == (2, 2, 5, 8)


def test_cached_decoding_matches_full_forward():
    torch.manual_seed(7)
    model = GPT(config=tiny_config()).eval()
    tokens = torch.randint(0, model.vocab_size, (2, 12))

    full_logits = model(tokens)
    prefix_logits, cache = model(tokens[:, :7], use_cache=True)
    pieces = [prefix_logits]
    for position in range(7, tokens.size(1)):
        logits, cache = model(
            tokens[:, position : position + 1], kv_cache=cache, use_cache=True
        )
        pieces.append(logits)

    cached_logits = torch.cat(pieces, dim=1)
    torch.testing.assert_close(cached_logits, full_logits, atol=2e-5, rtol=2e-5)


def test_sdpa_matches_manual_attention():
    torch.manual_seed(11)
    sdpa = GPT(config=tiny_config(use_sdpa=True)).eval()
    manual = GPT(config=tiny_config(use_sdpa=False)).eval()
    manual.load_state_dict(sdpa.state_dict())
    tokens = torch.randint(0, sdpa.vocab_size, (2, 10))
    torch.testing.assert_close(
        sdpa(tokens), manual(tokens), atol=2e-5, rtol=2e-5
    )


def test_invalid_gqa_ratio_is_rejected():
    try:
        MultiHeadSelfAttention(embed_dim=32, num_heads=4, num_kv_heads=3)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("invalid GQA configuration was accepted")


def test_sliding_window_bounds_kv_cache_and_preserves_absolute_position():
    attention = MultiHeadSelfAttention(
        embed_dim=32,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.0,
        position_encoding="rope",
        fused_qkv=True,
        sliding_window=4,
    ).eval()
    cache = None
    for _ in range(12):
        _, cache = attention(torch.randn(1, 1, 32), kv_cache=cache, use_cache=True)
    assert cache[0].size(2) == 4
    assert cache[1].size(2) == 4
    assert cache[2] == 12


def test_sliding_cached_decoding_matches_full_forward():
    torch.manual_seed(17)
    model = GPT(
        config=tiny_config(
            num_layers=1,
            sliding_window=4,
            global_attention_interval=0,
        )
    ).eval()
    tokens = torch.randint(0, model.vocab_size, (1, 12))
    expected = model(tokens)
    pieces = []
    cache = None
    for position in range(tokens.size(1)):
        logits, cache = model(
            tokens[:, position : position + 1], kv_cache=cache, use_cache=True
        )
        pieces.append(logits)
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), expected, atol=2e-5, rtol=2e-5
    )


def test_periodic_global_attention_layers_are_configured():
    model = GPT(
        config=tiny_config(
            num_layers=4,
            sliding_window=8,
            global_attention_interval=2,
        )
    )
    assert [layer.attention.sliding_window for layer in model.layers] == [8, None, 8, None]
