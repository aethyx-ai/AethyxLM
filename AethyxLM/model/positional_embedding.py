"""
Learnable Positional Embedding Layer for AethyxLM.
"""

import torch
import torch.nn as nn

from .config import CONTEXT_LENGTH, EMBED_DIM


class PositionalEmbedding(nn.Module):
    """
    Learns a unique embedding vector for each position in the sequence.
    """

    def __init__(
        self,
        context_length: int = CONTEXT_LENGTH,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embed_dim,
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings:
                Shape -> (batch_size, sequence_length, embed_dim)

        Returns:
            Positional embeddings:
                Shape -> (batch_size, sequence_length, embed_dim)
        """

        batch_size, sequence_length, _ = token_embeddings.shape

        positions = torch.arange(
            sequence_length,
            device=token_embeddings.device,
        )

        position_embeddings = self.embedding(positions)

        # Expand from (sequence_length, embed_dim)
        # to (batch_size, sequence_length, embed_dim)
        position_embeddings = position_embeddings.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        return position_embeddings