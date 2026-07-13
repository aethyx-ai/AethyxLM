"""
Learning Rate Scheduler with Warmup and Cosine Decay.
"""

import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Learning rate schedule with linear warmup and cosine decay.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Steps for linear warmup
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as fraction of peak LR (default 0.1)
        last_epoch: Last epoch index
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Constant LR with linear warmup.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Steps for linear warmup
        last_epoch: Last epoch index
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)