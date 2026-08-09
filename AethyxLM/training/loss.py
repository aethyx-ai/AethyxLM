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
    
    def __init__(self, ignore_index: int = -100, z_loss_coefficient: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.z_loss_coefficient = z_loss_coefficient
    
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
        logits = logits.reshape(batch_size * seq_len, vocab_size)
        targets = targets.reshape(batch_size * seq_len)
        
        loss = self.criterion(logits, targets)
        if self.z_loss_coefficient:
            valid = targets != self.ignore_index
            if valid.any():
                log_z = torch.logsumexp(logits[valid].float(), dim=-1)
                loss = loss + self.z_loss_coefficient * log_z.square().mean()
        return loss
