"""
Common neural network layers used throughout AethyxLM.

This module provides reusable building blocks that can be extended later
without changing the rest of the codebase.
"""

import torch.nn as nn


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
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )