"""
AethyxLM Dataset

Converts raw text into training samples.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer.tokenizer import AethyxTokenizer


class AethyxDataset(Dataset):

    def __init__(self, text_path, context_length=128):

        self.tokenizer = AethyxTokenizer()

        text_path = Path(text_path)

        if not text_path.exists():
            raise FileNotFoundError(text_path)

        text = text_path.read_text(
            encoding="utf-8"
        )

        self.tokens = self.tokenizer.encode(text)

        self.context_length = context_length

    def __len__(self):

        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):

        x = self.tokens[
            idx: idx + self.context_length
        ]

        y = self.tokens[
            idx + 1: idx + self.context_length + 1
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )