"""
AethyxLM Dataset

Converts raw text into training samples.
Supports .txt, .csv, and .json files.
"""

from pathlib import Path
import json
import csv
import random

import torch
from torch.utils.data import Dataset

from tokenizer.tokenizer import AethyxTokenizer


def read_csv_file(path: Path) -> str:
    """Read text from .csv file, auto-detecting text column."""
    text_parts = []
    
    with path.open('r', encoding='utf-8') as f:
        # Sniff dialect
        sample = f.read(1024)
        f.seek(0)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
        except csv.Error:
            dialect = csv.excel
        
        f.seek(0)
        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = reader.fieldnames or []
        
        # Prefer common text column names
        text_column = None
        for candidate in ['story', 'text', 'content', 'response', 'prompt', 'completion']:
            if candidate in fieldnames:
                text_column = candidate
                break
        
        if text_column is None and fieldnames:
            text_column = fieldnames[0]
        
        if text_column is None:
            raise ValueError(f"No text column found in CSV: {path}")
        
        for row in reader:
            text = row.get(text_column, '').strip()
            if text:
                text_parts.append(text)
    
    return "\n\n".join(text_parts)


def read_json_file(path: Path) -> str:
    """Read text from .json file, auto-detecting text field."""
    text_parts = []
    
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                text_parts.append(item.strip())
            elif isinstance(item, dict):
                for key in ['story', 'text', 'content', 'response', 'prompt', 'completion']:
                    if key in item and isinstance(item[key], str):
                        text_parts.append(item[key].strip())
                        break
    elif isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], list):
            # Handle nested data array
            for item in data['data']:
                if isinstance(item, str):
                    text_parts.append(item.strip())
                elif isinstance(item, dict):
                    for key in ['story', 'text', 'content', 'response', 'prompt', 'completion']:
                        if key in item and isinstance(item[key], str):
                            text_parts.append(item[key].strip())
                            break
        else:
            for key in ['story', 'text', 'content', 'response', 'prompt', 'completion']:
                if key in data and isinstance(data[key], str):
                    text_parts.append(data[key].strip())
    
    return "\n\n".join(text_parts)


def read_text_file(path: Path) -> str:
    """Read text from file based on extension."""
    suffix = path.suffix.lower()
    
    if suffix == '.txt':
        return path.read_text(encoding="utf-8")
    elif suffix == '.csv':
        return read_csv_file(path)
    elif suffix == '.json':
        return read_json_file(path)
    else:
        return path.read_text(encoding="utf-8")


class AethyxDataset(Dataset):

    def __init__(self, text_path, context_length=128, seed: int = 42):

        self.tokenizer = AethyxTokenizer()

        text_path = Path(text_path)

        if not text_path.exists():
            raise FileNotFoundError(text_path)

        text = read_text_file(text_path)

        self.tokens = self.tokenizer.encode(text)

        self.context_length = context_length

        # Deterministic seed for reproducibility
        random.seed(seed)

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