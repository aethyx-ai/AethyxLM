import pytest
import torch.nn as nn

from inference.quantization import quantization_coverage, quantize_model_for_inference
from model.layers import Linear


def test_quantization_is_explicit_and_cpu_only():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    assert quantize_model_for_inference(model, "none", "cpu") is model
    with pytest.raises(ValueError, match="CPU-only"):
        quantize_model_for_inference(model, "dynamic-int8", "cuda")
    with pytest.raises(ValueError, match="Unsupported"):
        quantize_model_for_inference(model, "int4", "cpu")


def test_dynamic_quantization_covers_custom_linear_subclasses():
    model = nn.Sequential(Linear(4, 8), nn.ReLU(), Linear(8, 2))
    quantized = quantize_model_for_inference(model, "dynamic-int8", "cpu")
    assert quantization_coverage(quantized) == {
        "float_linear_modules": 0,
        "quantized_linear_modules": 2,
    }
