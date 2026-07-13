"""
Cross-Entropy Loss for Language Modeling.
"""

import torch
import torch.nn as nn


class LanguageModelLoss(nn.Module):
    """
    Cross-entropy loss for next-token prediction.
    
    Handles label shifting internally:
    - input_ids: (batch, seq_len) - input tokens
    - targets: (batch, seq_len) - target tokens (shifted by 1)
    """
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: (batch, seq_len, vocab_size) - model output
            targets: (batch, seq_len) - target token ids
            
        Returns:
            Scalar loss tensor
        """
        # Flatten for CrossEntropyLoss: (batch * seq_len, vocab_size)
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(batch_size * seq_len, vocab_size)
        targets = targets.view(batch_size * seq_len)
        
        loss = self.criterion(logits, targets)
        return loss