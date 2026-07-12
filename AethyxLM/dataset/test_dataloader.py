"""
AethyxLM DataLoader Test
"""

from .dataloader import create_dataloader


def main():

    loader = create_dataloader()

    print("=" * 60)

    print(f"Number of batches: {len(loader)}")

    x, y = next(iter(loader))

    print("\nInput Batch Shape:")
    print(x.shape)

    print("\nTarget Batch Shape:")
    print(y.shape)

    print("\nFirst Input Sequence:")
    print(x[0])

    print("\nFirst Target Sequence:")
    print(y[0])

    print("=" * 60)


if __name__ == "__main__":
    main()