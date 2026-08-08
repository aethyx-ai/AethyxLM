import torch

from model.gpt import GPT
from model.modules.rmsnorm import RMSNorm


def test_modern_config_wires_all_components():
    config = {
        "vocab_size": 128,
        "context_length": 32,
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 128,
        "dropout": 0.0,
        "normalization": "rmsnorm",
        "position_encoding": "rope",
        "ffn_type": "swiglu",
        "rope_base": 500000.0,
        "rope_max_seq_len": 256,
    }

    model = GPT(config=config)

    assert model.position_embedding is None
    assert isinstance(model.final_norm, RMSNorm)
    for layer in model.layers:
        assert isinstance(layer.norm1, RMSNorm)
        assert isinstance(layer.norm2, RMSNorm)
        assert layer.feed_forward.ffn_type == "swiglu"
        assert layer.attention.position_encoding == "rope"
        assert layer.attention.rope is not None

    output = model(torch.randint(0, config["vocab_size"], (2, 16)))
    assert output.shape == (2, 16, config["vocab_size"])
