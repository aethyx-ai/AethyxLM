"""
Test the complete GPT model.
"""

import torch

from dataset.dataloader import create_dataloader

from model.gpt import GPT


def main():
    print("=" * 60)
    print("Testing Complete GPT Model")
    print("=" * 60)

    # --------------------------------------------------
    # Load one batch
    # --------------------------------------------------

    dataloader = create_dataloader()

    input_batch, target_batch = next(iter(dataloader))

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = GPT()

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    logits = model(input_batch)

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    print(f"Input Shape :  {input_batch.shape}")
    print(f"Target Shape:  {target_batch.shape}")
    print()

    print(f"Output Shape: {logits.shape}")
    print()

    print(f"Contains NaN : {torch.isnan(logits).any().item()}")
    print(f"Contains Inf : {torch.isinf(logits).any().item()}")
    print()

    print("Sample Logits:")
    print(logits[0, 0, :20])

    print()

    print("=" * 60)
    print("✓ GPT Model passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()