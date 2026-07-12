"""
Token Embedding Layer for AethyxLM.
"""

import torch
import torch.nn as nn

from .config import VOCAB_SIZE, EMBED_DIM


class TokenEmbedding(nn.Module):
    """
    Converts token IDs into dense embedding vectors.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids:
                Shape -> (batch_size, sequence_length)

        Returns:
            Shape -> (batch_size, sequence_length, embed_dim)
        """

        return self.embedding(token_ids)