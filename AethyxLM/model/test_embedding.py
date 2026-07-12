"""
Test script for the Token Embedding layer.
"""

import torch

from dataset.dataloader import create_dataloader
from model.embedding import TokenEmbedding


def main():
    dataloader = create_dataloader()

    input_batch, _ = next(iter(dataloader))

    embedding = TokenEmbedding()

    output = embedding(input_batch)

    print("=" * 60)

    print("Input Shape:")
    print(input_batch.shape)

    print()

    print("Output Shape:")
    print(output.shape)

    print()

    print("Embedding Dimension:")
    print(output.shape[-1])

    print()

    print("Sample Embedding Vector:")
    print(output[0, 0])

    print("=" * 60)


if __name__ == "__main__":
    main()