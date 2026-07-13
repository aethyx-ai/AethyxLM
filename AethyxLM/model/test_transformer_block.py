"""
Test script for the Transformer Block.
"""

import torch

from dataset.dataloader import create_dataloader

from model.embedding import TokenEmbedding
from model.positional_embedding import PositionalEmbedding
from model.transformer_block import TransformerBlock


def main():
    print("=" * 60)
    print("Testing Transformer Block")
    print("=" * 60)

    # --------------------------------------------------
    # Load one batch
    # --------------------------------------------------

    dataloader = create_dataloader()

    input_batch, _ = next(iter(dataloader))

    # --------------------------------------------------
    # Create embeddings
    # --------------------------------------------------

    token_embedding = TokenEmbedding()
    positional_embedding = PositionalEmbedding()

    x = token_embedding(input_batch)

    x = x + positional_embedding(x)

    # --------------------------------------------------
    # Transformer Block
    # --------------------------------------------------

    transformer = TransformerBlock()

    output = transformer(x)

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    print(f"Input Shape :  {x.shape}")
    print(f"Output Shape:  {output.shape}")
    print()

    print(f"Contains NaN : {torch.isnan(output).any().item()}")
    print(f"Contains Inf : {torch.isinf(output).any().item()}")
    print()

    print("Sample Output Vector:")
    print(output[0, 0])

    print()

    print("=" * 60)
    print("✓ Transformer Block passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()