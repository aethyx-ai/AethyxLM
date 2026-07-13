"""
Test script for the Positional Embedding layer.
"""

from dataset.dataloader import create_dataloader
from model.embedding import TokenEmbedding
from model.positional_embedding import PositionalEmbedding


def main():
    dataloader = create_dataloader()

    input_batch, _ = next(iter(dataloader))

    token_embedding = TokenEmbedding()
    positional_embedding = PositionalEmbedding()

    token_vectors = token_embedding(input_batch)
    position_vectors = positional_embedding(token_vectors)

    combined = token_vectors + position_vectors

    print("=" * 60)

    print("Token Embedding Shape:")
    print(token_vectors.shape)

    print()

    print("Position Embedding Shape:")
    print(position_vectors.shape)

    print()

    print("Combined Shape:")
    print(combined.shape)

    print()

    print("First Position Vector:")
    print(position_vectors[0, 0])

    print("=" * 60)


if __name__ == "__main__":
    main()