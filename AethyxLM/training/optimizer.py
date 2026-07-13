"""
AdamW Optimizer for AethyxLM.
"""

import torch
from torch.optim import AdamW


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """
    Create AdamW optimizer with weight decay.
    
    Applies weight decay only to weights, not biases or LayerNorm params.
    
    Args:
        model: The model to optimize
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        eps: Epsilon for numerical stability
        
    Returns:
        Configured AdamW optimizer
    """
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "bias" in name or "ln" in name.lower() or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )
    
    return optimizer