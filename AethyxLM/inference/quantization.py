"""Conservative, opt-in inference quantization helpers."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from model.layers import Linear


QUANTIZATION_MODES = ("none", "dynamic-int8-ffn", "dynamic-int8")


def quantization_coverage(model: nn.Module) -> dict[str, int]:
    """Report projected layers before/after quantization by executable type."""
    float_linears = sum(isinstance(module, nn.Linear) for module in model.modules())
    quantized_linears = sum(
        module.__class__.__module__.startswith("torch.ao.nn.quantized")
        and module.__class__.__name__ == "Linear"
        for module in model.modules()
    )
    return {
        "float_linear_modules": float_linears,
        "quantized_linear_modules": quantized_linears,
    }


def quantize_model_for_inference(model: nn.Module, mode: str, device: str):
    """Apply a supported inference-only quantizer with explicit device guards."""
    if mode == "none":
        return model
    if mode not in {"dynamic-int8", "dynamic-int8-ffn"}:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    if device != "cpu":
        raise ValueError("dynamic-int8 quantization is CPU-only; use --device cpu")
    candidate = copy.deepcopy(model)

    def canonicalize(module):
        for name, child in list(module.named_children()):
            if type(child) is Linear:
                replacement = nn.Linear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                replacement.weight = child.weight
                if child.bias is not None:
                    replacement.bias = child.bias
                setattr(module, name, replacement)
            else:
                canonicalize(child)

    canonicalize(candidate)
    if mode == "dynamic-int8-ffn":
        qconfig_spec = {
            name: torch.ao.quantization.default_dynamic_qconfig
            for name, module in candidate.named_modules()
            if ".feed_forward." in name and isinstance(module, nn.Linear)
        }
    else:
        qconfig_spec = {nn.Linear}
    return torch.ao.quantization.quantize_dynamic(
        candidate,
        qconfig_spec,
        dtype=torch.qint8,
        inplace=True,
    )
