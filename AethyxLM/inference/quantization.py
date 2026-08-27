"""Conservative, opt-in inference quantization helpers."""

from __future__ import annotations

import torch
import torch.nn as nn


QUANTIZATION_MODES = ("none", "dynamic-int8")


def quantize_model_for_inference(model: nn.Module, mode: str, device: str):
    """Apply a supported inference-only quantizer with explicit device guards."""
    if mode == "none":
        return model
    if mode != "dynamic-int8":
        raise ValueError(f"Unsupported quantization mode: {mode}")
    if device != "cpu":
        raise ValueError("dynamic-int8 quantization is CPU-only; use --device cpu")
    return torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )
