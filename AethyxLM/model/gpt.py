"""
Complete GPT model for AethyxLM.
"""

import torch
import torch.nn as nn

from .config import (
    VOCAB_SIZE,
    EMBED_DIM,
    NUM_LAYERS,
)

from .embedding import TokenEmbedding
from .positional_embedding import PositionalEmbedding
from .transformer_block import TransformerBlock
from .layer_norm import LayerNorm


class GPT(nn.Module):
    """
    Decoder-only GPT Language Model.
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------
        # Token Embedding
        # ----------------------------------------

        self.token_embedding = TokenEmbedding()

        # ----------------------------------------
        # Positional Embedding
        # ----------------------------------------

        self.position_embedding = PositionalEmbedding()

        # ----------------------------------------
        # Transformer Blocks
        # ----------------------------------------

        self.layers = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(NUM_LAYERS)
            ]
        )

        # ----------------------------------------
        # Final Layer Normalization
        # ----------------------------------------

        self.final_norm = LayerNorm(EMBED_DIM)

        # ----------------------------------------
        # Language Modeling Head
        # ----------------------------------------

        self.lm_head = nn.Linear(
            EMBED_DIM,
            VOCAB_SIZE,
            bias=False,
        )

        #Share embedding weights with output layer
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids:
                Shape -> (batch_size, context_length)

        Returns:
            logits:
                Shape -> (batch_size, context_length, vocab_size)
        """

        # ----------------------------------------
        # Token Embeddings
        # ----------------------------------------

        x = self.token_embedding(input_ids)

        # ----------------------------------------
        # Add Positional Embeddings
        # ----------------------------------------

        x = x + self.position_embedding(x)

        # ----------------------------------------
        # Transformer Blocks
        # ----------------------------------------

        for layer in self.layers:
            x = layer(x)

        # ----------------------------------------
        # Final LayerNorm
        # ----------------------------------------

        x = self.final_norm(x)

        # ----------------------------------------
        # Project to Vocabulary
        # ----------------------------------------

        logits = self.lm_head(x)

        return logits