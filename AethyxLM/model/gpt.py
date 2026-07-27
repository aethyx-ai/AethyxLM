"""
Complete GPT model for AethyxLM.
"""

import torch
import torch.nn as nn

from .config import (
    EMBED_DIM,
    NUM_LAYERS,
)


class GPT(nn.Module):
    """
    Decoder-only GPT Language Model.
    """

    def __init__(self, vocab_size: int = None):
        super().__init__()

        if vocab_size is None:
            from .config import VOCAB_SIZE
            vocab_size = VOCAB_SIZE

        # ----------------------------------------
        # Token Embedding
        # ----------------------------------------

        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)

        # ----------------------------------------
        # Positional Embedding
        # ----------------------------------------

        self.position_embedding = nn.Embedding(128, EMBED_DIM)

        # ----------------------------------------
        # Transformer Blocks
        # ----------------------------------------

        from .transformer_block import TransformerBlock
        self.layers = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(NUM_LAYERS)
            ]
        )

        # ----------------------------------------
        # Final Layer Normalization
        # ----------------------------------------

        from .layer_norm import LayerNorm
        self.final_norm = LayerNorm(EMBED_DIM)

        # ----------------------------------------
        # Language Modeling Head
        # ----------------------------------------

        self.lm_head = nn.Linear(
            EMBED_DIM,
            vocab_size,
            bias=False,
        )

        #Share embedding weights with output layer
        self.lm_head.weight = self.token_embedding.weight

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

        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        x = x + self.position_embedding(positions)

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