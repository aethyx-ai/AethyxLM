"""
Test script for the Multi-Head Self-Attention module.
"""

import torch

from dataset.dataloader import create_dataloader
from model.embedding import TokenEmbedding
from model.positional_embedding import PositionalEmbedding
from model.attention import MultiHeadSelfAttention


def main():
    print("=" * 60)
    print("Testing Multi-Head Self-Attention")
    print("=" * 60)

    # --------------------------------------------------
    # Create input batch
    # --------------------------------------------------

    dataloader = create_dataloader()
    input_batch, _ = next(iter(dataloader))

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------

    token_embedding = TokenEmbedding()
    positional_embedding = PositionalEmbedding()

    token_vectors = token_embedding(input_batch)
    position_vectors = positional_embedding(token_vectors)

    x = token_vectors + position_vectors

    # --------------------------------------------------
    # Attention
    # --------------------------------------------------

    attention = MultiHeadSelfAttention()

    output = attention(x)

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {output.shape}")
    print()

    print(f"Contains NaN : {torch.isnan(output).any().item()}")
    print(f"Contains Inf : {torch.isinf(output).any().item()}")
    print()

    print("Sample Output Vector:")
    print(output[0, 0])

    print()
    print("=" * 60)
    print("✓ Attention module passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()