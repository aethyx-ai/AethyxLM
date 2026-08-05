"""
Common neural network layers used throughout AethyxLM.

This module provides reusable building blocks that can be extended later
without changing the rest of the codebase.
"""

import torch
import torch.nn as nn
import math


def init_weights(module: nn.Module, init_type: str = "normal", init_std: float = 0.02):
    """
    Initialize weights for a module.

    Args:
        module: The module to initialize.
        init_type: Initialization type ("normal", "xavier", "kaiming", "orthogonal").
        init_std: Standard deviation for normal initialization.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        if init_type == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
        elif init_type == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif init_type == "kaiming":
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        elif init_type == "orthogonal":
            nn.init.orthogonal_(module.weight)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")

        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.LayerNorm,)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    # Handle our custom RMSNorm
    from model.modules.rmsnorm import RMSNorm
    if isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)


def init_module(module: nn.Module, init_type: str = "normal", init_std: float = 0.02):
    """
    Recursively initialize all weights in a module.

    Args:
        module: The module to initialize.
        init_type: Initialization type ("normal", "xavier", "kaiming", "orthogonal").
        init_std: Standard deviation for normal initialization.
    """
    for m in module.modules():
        init_weights(m, init_type, init_std)


class Linear(nn.Linear):
    """
    Standard Linear layer.

    This currently behaves exactly like PyTorch's nn.Linear, but having our own
    wrapper allows us to later add:

    - Custom weight initialization
    - Quantization
    - LoRA adapters
    - Profiling
    - Logging
    - Optimized kernels

    without modifying every module individually.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_type: str = "normal",
        init_std: float = 0.02,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        # Apply custom initialization
        init_weights(self, init_type, init_std)