import pytest
import torch.nn as nn

from inference.quantization import quantize_model_for_inference


def test_quantization_is_explicit_and_cpu_only():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    assert quantize_model_for_inference(model, "none", "cpu") is model
    with pytest.raises(ValueError, match="CPU-only"):
        quantize_model_for_inference(model, "dynamic-int8", "cuda")
    with pytest.raises(ValueError, match="Unsupported"):
        quantize_model_for_inference(model, "int4", "cpu")
