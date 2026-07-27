"""
Random seed utility for reproducibility.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Allow cuDNN auto-tuner for better performance
    torch.backends.cudnn.benchmark = True
    # Note: For exact reproducibility, set benchmark=False and deterministic=True
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False