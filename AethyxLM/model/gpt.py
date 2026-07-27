"""
Complete GPT model for AethyxLM.
"""

import torch
import torch.nn as nn

from model.config import (
    EMBED_DIM,
    NUM_LAYERS,
    CONTEXT_LENGTH,
    DROPOUT,
)


class GPT(nn.Module):
    """
    Decoder-only GPT Language Model.
    """

    def __init__(self, vocab_size: int = None, config: dict = None):
        super().__init__()

        # Merge defaults from config.py with overrides from config dict
        from model.config import (
            VOCAB_SIZE as DEFAULT_VOCAB_SIZE,
            CONTEXT_LENGTH as DEFAULT_CONTEXT_LENGTH,
            EMBED_DIM as DEFAULT_EMBED_DIM,
            NUM_HEADS as DEFAULT_NUM_HEADS,
            NUM_LAYERS as DEFAULT_NUM_LAYERS,
            FFN_DIM as DEFAULT_FFN_DIM,
            DROPOUT as DEFAULT_DROPOUT,
            USE_BIAS as DEFAULT_USE_BIAS,
            LAYER_NORM_EPS as DEFAULT_LAYER_NORM_EPS,
        )

        if config is None:
            config = {}

        self.vocab_size = vocab_size if vocab_size is not None else config.get('vocab_size', DEFAULT_VOCAB_SIZE)
        self.context_length = config.get('context_length', DEFAULT_CONTEXT_LENGTH)
        self.embed_dim = config.get('embed_dim', DEFAULT_EMBED_DIM)
        self.num_heads = config.get('num_heads', DEFAULT_NUM_HEADS)
        self.num_layers = config.get('num_layers', DEFAULT_NUM_LAYERS)
        self.ffn_dim = config.get('ffn_dim', DEFAULT_FFN_DIM)
        self.dropout_rate = config.get('dropout', DEFAULT_DROPOUT)
        self.use_bias = config.get('use_bias', DEFAULT_USE_BIAS)
        self.layer_norm_eps = config.get('layer_norm_eps', DEFAULT_LAYER_NORM_EPS)

        # ----------------------------------------
        # Token Embedding
        # ----------------------------------------

        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # ----------------------------------------
        # Positional Embedding
        # ----------------------------------------

        self.position_embedding = nn.Embedding(self.context_length, self.embed_dim)

        # ----------------------------------------
        # Dropout
        # ----------------------------------------

        self.dropout = nn.Dropout(self.dropout_rate)

        # ----------------------------------------
        # Transformer Blocks
        # ----------------------------------------

        from model.transformer_block import TransformerBlock
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=self.dropout_rate,
                    context_length=self.context_length,
                    use_bias=self.use_bias,
                    layer_norm_eps=self.layer_norm_eps,
                )
                for _ in range(self.num_layers)
            ]
        )

        # ----------------------------------------
        # Final Layer Normalization
        # ----------------------------------------

        from model.layer_norm import LayerNorm
        self.final_norm = LayerNorm(self.embed_dim, eps=self.layer_norm_eps)

        # ----------------------------------------
        # Language Modeling Head
        # ----------------------------------------

        self.lm_head = nn.Linear(
            self.embed_dim,
            self.vocab_size,
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
        x = self.position_embedding(positions) + x
        x = self.dropout(x)

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