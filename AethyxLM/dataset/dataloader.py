"""
AethyxLM DataLoader

Creates PyTorch DataLoaders for training.
"""

from torch.utils.data import DataLoader

from dataset.config import (
    DATA_FILE,
    CONTEXT_LENGTH,
    BATCH_SIZE,
    NUM_WORKERS,
    SHUFFLE,
)
from dataset.dataset import AethyxDataset


def create_dataloader():

    dataset = AethyxDataset(
        text_path=DATA_FILE,
        context_length=CONTEXT_LENGTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    return loader