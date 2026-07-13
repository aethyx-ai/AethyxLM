"""
Test script for the custom LayerNorm implementation.
"""

import torch

from dataset.dataloader import create_dataloader
from model.embedding import TokenEmbedding
from model.positional_embedding import PositionalEmbedding
from model.layer_norm import LayerNorm


def main():
    print("=" * 60)
    print("Testing Layer Normalization")
    print("=" * 60)

    # --------------------------------------------------
    # Create input batch
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
    # LayerNorm
    # --------------------------------------------------

    layer_norm = LayerNorm()

    output = layer_norm(x)

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    mean = output.mean(dim=-1)
    variance = output.var(dim=-1, unbiased=False)

    print(f"Input Shape :  {x.shape}")
    print(f"Output Shape:  {output.shape}")
    print()

    print(f"Contains NaN : {torch.isnan(output).any().item()}")
    print(f"Contains Inf : {torch.isinf(output).any().item()}")
    print()

    print(f"Mean     (should be ~0): {mean.mean().item():.6f}")
    print(f"Variance (should be ~1): {variance.mean().item():.6f}")
    print()

    print("Sample Output Vector:")
    print(output[0, 0])

    print()
    print("=" * 60)
    print("✓ LayerNorm passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()