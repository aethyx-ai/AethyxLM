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
    """
    Memory-efficient dataset that stores tokenised data as a memory-mapped
    numpy uint16 array (.bin file) instead of keeping everything in RAM.

    On first run it reads the .txt file, tokenises it, and saves a companion
    `<name>.bin` file alongside the text file.  Subsequent runs skip
    tokenisation and mmap the .bin directly — startup goes from ~30 s to <1 s
    and RAM usage stays constant regardless of dataset size.
    """

    def __init__(self, text_path, context_length=128, seed: int = 42):
        import numpy as np

        self.context_length = context_length
        text_path = Path(text_path)

        if not text_path.exists():
            raise FileNotFoundError(text_path)

        bin_path = text_path.with_suffix('.bin')

        if not bin_path.exists():
            # --- First-time tokenisation ---
            print(f"Tokenising {text_path} -> {bin_path} ...")
            tokenizer = AethyxTokenizer()
            text = read_text_file(text_path)
            token_ids = tokenizer.encode(text)
            arr = np.array(token_ids, dtype=np.uint16)
            arr.tofile(bin_path)
            print(f"[OK] Saved {len(arr):,} tokens to {bin_path}")
            del text, token_ids, arr

        # --- Memory-map the .bin file ---
        self._data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        print(f"[OK] mmap {bin_path}: {len(self._data):,} tokens")

        random.seed(seed)

    def __len__(self):
        return max(0, len(self._data) - self.context_length)

    def __getitem__(self, idx):
        import numpy as np
        chunk = np.array(self._data[idx: idx + self.context_length + 1],
                         dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y