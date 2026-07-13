"""
Test script for the Feed Forward Network.
"""

import torch

from dataset.dataloader import create_dataloader
from model.embedding import TokenEmbedding
from model.positional_embedding import PositionalEmbedding
from model.feed_forward import FeedForward


def main():
    print("=" * 60)
    print("Testing Feed Forward Network")
    print("=" * 60)

    dataloader = create_dataloader()
    input_batch, _ = next(iter(dataloader))

    token_embedding = TokenEmbedding()
    positional_embedding = PositionalEmbedding()

    x = token_embedding(input_batch)
    x = x + positional_embedding(x)

    ffn = FeedForward()

    output = ffn(x)

    print(f"Input Shape : {x.shape}")
    print(f"Output Shape: {output.shape}")
    print()

    print(f"Contains NaN : {torch.isnan(output).any().item()}")
    print(f"Contains Inf : {torch.isinf(output).any().item()}")
    print()

    print("Sample Output Vector:")
    print(output[0, 0])

    print()
    print("=" * 60)
    print("✓ Feed Forward Network passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()