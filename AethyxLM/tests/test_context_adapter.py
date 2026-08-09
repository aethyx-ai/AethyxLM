import torch

from model.context_adapter import LatentContextAdapter
from model.gpt import GPT


def context_model():
    return GPT(
        config={
            "vocab_size": 101,
            "context_length": 32,
            "embed_dim": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "num_layers": 3,
            "ffn_dim": 64,
            "dropout": 0.0,
            "normalization": "rmsnorm",
            "position_encoding": "rope",
            "ffn_type": "swiglu",
            "fused_qkv": True,
            "context_adapter": {
                "enabled": True,
                "input_dim": 24,
                "num_latents": 8,
                "num_heads": 4,
                "num_types": 6,
                "depth": 1,
                "cross_attention_layers": [1, 2],
            },
        }
    )


def test_adapter_enforces_a_fixed_context_budget():
    adapter = LatentContextAdapter(24, 32, num_latents=8, num_heads=4)
    short = adapter(torch.randn(2, 10, 24))
    long = adapter(torch.randn(2, 100, 24))
    assert short.shape == long.shape == (2, 8, 32)


def test_typed_masked_context_changes_decoder_output():
    torch.manual_seed(5)
    model = context_model().eval()
    tokens = torch.randint(0, 101, (2, 7))
    features = torch.randn(2, 12, 24)
    types = torch.randint(0, 6, (2, 12))
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[:, -3:] = False

    baseline = model(tokens)
    conditioned = model(
        tokens,
        context_features=features,
        context_type_ids=types,
        context_mask=mask,
    )
    assert conditioned.shape == baseline.shape
    assert not torch.allclose(conditioned, baseline)


def test_preencoded_context_can_be_reused_during_cached_decoding():
    model = context_model().eval()
    features = torch.randn(1, 20, 24)
    latents = model.encode_context(features)
    prefix = torch.randint(0, 101, (1, 5))
    logits, cache = model(prefix, context_latents=latents, use_cache=True)
    next_logits, cache = model(
        torch.randint(0, 101, (1, 1)),
        kv_cache=cache,
        use_cache=True,
        context_latents=latents,
    )
    assert logits.shape == (1, 5, 101)
    assert next_logits.shape == (1, 1, 101)
    assert cache[0][0].size(2) == 6
