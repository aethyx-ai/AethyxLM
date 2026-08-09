import torch

from model.gpt import GPT


def config():
    return {
        "vocab_size": 67,
        "context_length": 16,
        "embed_dim": 32,
        "num_heads": 4,
        "num_kv_heads": 2,
        "num_layers": 2,
        "ffn_dim": 64,
        "dropout": 0.0,
        "normalization": "rmsnorm",
        "position_encoding": "rope",
        "ffn_type": "swiglu",
        "fused_qkv": True,
        "use_sdpa": True,
        "sliding_window": 8,
    }


def test_fused_gqa_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(23)
    original = GPT(config=config()).eval()
    tokens = torch.randint(0, 67, (2, 10))
    expected = original(tokens)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": original.state_dict(),
            "config": {"model": config()},
        },
        checkpoint,
    )
    restored = GPT.from_checkpoint(checkpoint).eval()
    torch.testing.assert_close(restored(tokens), expected)


def test_legacy_persistent_causal_masks_are_ignored():
    model = GPT(
        config={
            "vocab_size": 32,
            "context_length": 8,
            "embed_dim": 16,
            "num_heads": 4,
            "num_layers": 1,
            "ffn_dim": 32,
        }
    )
    legacy_state = dict(model.state_dict())
    legacy_state["layers.0.attention.causal_mask"] = torch.ones(8, 8)
    model.load_compatible_state_dict(legacy_state, strict=True)


def test_legacy_flat_stale_metadata_is_reconstructed_from_weights(tmp_path):
    legacy_config = {
        "vocab_size": 41,
        "context_length": 8,
        "embed_dim": 16,
        "num_heads": 4,
        "num_layers": 1,
        "ffn_dim": 32,
        "dropout": 0.1,
        "normalization": "layernorm",
        "position_encoding": "learned",
        "ffn_type": "gelu",
    }
    original = GPT(config=legacy_config).eval()
    state = dict(original.state_dict())
    state["layers.0.attention.causal_mask"] = torch.ones(8, 8)
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": state,
            "config": {"vocab_size": 32000, "context_length": 8, "num_layers": 1},
        },
        checkpoint,
    )
    restored = GPT.from_checkpoint(checkpoint).eval()
    assert restored.vocab_size == 41
    assert restored.embed_dim == 16
    assert restored.ffn_dim == 32
    assert restored.position_encoding == "learned"
