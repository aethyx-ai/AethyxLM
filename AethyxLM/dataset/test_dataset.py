"""
AethyxLM Dataset Test

Tests whether the dataset correctly creates
input-target training pairs.
"""

from .config import DATA_FILE, CONTEXT_LENGTH
from .dataset import AethyxDataset


def main():

    dataset = AethyxDataset(
        text_path=DATA_FILE,
        context_length=CONTEXT_LENGTH,
    )

    print("=" * 60)

    print(f"Dataset Size: {len(dataset)}")

    x, y = dataset[0]

    print("\nInput Tensor:")
    print(x)

    print("\nTarget Tensor:")
    print(y)

    print("\nInput Shape:")
    print(x.shape)

    print("\nTarget Shape:")
    print(y.shape)

    print("=" * 60)


if __name__ == "__main__":
    main()