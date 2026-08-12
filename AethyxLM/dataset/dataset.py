"""
AethyxLM Dataset

Converts raw text into training samples.
Supports .txt, .csv, and .json files.
"""

from pathlib import Path
import json
import csv
import random
import os

import torch
import numpy as np
from torch.utils.data import Dataset, Sampler

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


def worker_init_fn(worker_id: int):
    """Initialize worker with unique seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DistributedStridedSampler(Sampler):
    """Partition a virtual dataset across ranks without materializing indices.

    ``DistributedSampler(shuffle=True)`` creates ``randperm(len(dataset))``.
    That is prohibitively large for the mixed dataset because its logical length
    can be hundreds of millions even though its sources are memory-mapped.  The
    mixed dataset already derives deterministic random samples from
    ``(seed, epoch, index)``, so a disjoint strided index range provides DDP
    sharding without a second global shuffle or an O(dataset length) allocation.
    """

    def __init__(self, dataset, num_replicas: int, rank: int):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = len(dataset) // self.num_replicas
        self.epoch = 0

    def __iter__(self):
        stop = self.rank + self.num_samples * self.num_replicas
        return iter(range(self.rank, stop, self.num_replicas))

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        # MixedAethyxDataset owns epoch-dependent randomization.  Retaining this
        # method keeps the standard Trainer sampler lifecycle compatible.
        self.epoch = int(epoch)


class AethyxDataset(Dataset):
    """
    Memory-efficient dataset that stores tokenised data as a memory-mapped
    numpy uint16 array (.bin file) instead of keeping everything in RAM.

    On first run it reads the .txt file, tokenises it, and saves a companion
    `<name>.bin` file alongside the text file.  Subsequent runs skip
    tokenisation and mmap the .bin directly — startup goes from ~30 s to <1 s
    and RAM usage stays constant regardless of dataset size.
    """

    def __init__(
        self, text_path, context_length=128, seed: int = 42, tokenizer_path=None
    ):
        self.context_length = context_length
        text_path = Path(text_path)

        if not text_path.exists():
            raise FileNotFoundError(text_path)

        bin_path = text_path.with_suffix('.bin')
        metadata_path = bin_path.with_suffix('.bin.meta.json')

        selected_tokenizer = (
            AethyxTokenizer(tokenizer_path) if tokenizer_path else AethyxTokenizer()
        )
        if bin_path.exists() and metadata_path.exists():
            cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached_hash = cache_metadata.get("tokenizer_sha256")
            if cached_hash and cached_hash != selected_tokenizer.sha256:
                raise ValueError(
                    f"Token cache {bin_path} was built with a different tokenizer; "
                    "remove the .bin cache and rebuild it."
                )

        if not bin_path.exists() or bin_path.stat().st_size == 0:
            # --- First-time tokenisation (streaming to avoid OOM) ---
            print(f"Tokenising {text_path} -> {bin_path} (streaming)...")
            tokenizer = selected_tokenizer
            CHUNK_SIZE = 10_000_000  # tokens per write
            total_tokens = 0
            with open(bin_path, 'wb') as f_out:
                with text_path.open('r', encoding='utf-8') as f_in:
                    buffer = []
                    document_lines = []

                    def append_document(lines):
                        if not lines:
                            return
                        ids = tokenizer.encode("\n".join(lines))
                        if tokenizer.eos_id is not None:
                            ids.append(tokenizer.eos_id)
                        buffer.extend(ids)

                    for line in f_in:
                        if line.strip():
                            document_lines.append(line.rstrip("\n"))
                        else:
                            append_document(document_lines)
                            document_lines = []
                        if len(buffer) >= CHUNK_SIZE:
                            arr = np.array(buffer[:CHUNK_SIZE], dtype=np.uint16)
                            arr.tofile(f_out)
                            total_tokens += len(arr)
                            buffer = buffer[CHUNK_SIZE:]
                    append_document(document_lines)
                    if buffer:
                        arr = np.array(buffer, dtype=np.uint16)
                        arr.tofile(f_out)
                        total_tokens += len(arr)
            print(f"[OK] Saved {total_tokens:,} tokens to {bin_path}")
            metadata_path.write_text(
                json.dumps(
                    {
                        "tokenizer_file": str(tokenizer.path),
                        "tokenizer_sha256": tokenizer.sha256,
                        "vocab_size": tokenizer.vocab_size,
                        "tokens": total_tokens,
                        "dtype": "uint16",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        # --- Memory-map the .bin file ---
        self._data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        print(f"[OK] mmap {bin_path}: {len(self._data):,} tokens")

        self.seed = seed
        self.epoch = 0
        random.seed(seed)

    def __len__(self):
        return max(0, len(self._data) - self.context_length)

    def __getitem__(self, idx):
        chunk = np.array(self._data[idx: idx + self.context_length + 1],
                         dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class MixedAethyxDataset(Dataset):
    """
    Memory-efficient mixed dataset that samples from multiple AethyxDatasets
    according to configurable weights.
    
    Each sub-dataset is an AethyxDataset backed by a memory-mapped .bin file.
    Sampling is done according to configurable weights.
    """

    def __init__(
        self, datasets_config: list, context_length=128, seed: int = 42, tokenizer_path=None
    ):
        """
        Args:
            datasets_config: List of dicts with keys:
                - 'train': path to train .bin file
                - 'val': path to val .bin file (optional)
                - 'weight': sampling weight (will be normalized)
            context_length: sequence length
            seed: random seed
        """
        self.context_length = context_length
        self.seed = seed
        self.epoch = 0
        random.seed(seed)
        np.random.seed(seed)

        # Load datasets config
        self.datasets_config = datasets_config
        self.sub_datasets = []
        self.weights = []

        # Load each sub-dataset
        for config in datasets_config:
            weight = config.get('weight', 1.0)
            train_path = config['train']
            train_dataset = AethyxDataset(
                train_path,
                context_length=context_length,
                tokenizer_path=tokenizer_path,
            )
            self.sub_datasets.append(train_dataset)
            self.weights.append(weight)

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        # Calculate cumulative weights for sampling
        self.cumulative_weights = np.cumsum(self.weights)

        # One logical epoch covers the aggregate amount of source data. Weights
        # control sampling frequency, not the declared epoch length.
        self.total_length = sum(len(ds) for ds in self.sub_datasets)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return int(self.total_length)

    def __getitem__(self, idx):
        # Derive randomness from index and epoch so behavior is reproducible
        # across worker counts and resume runs.
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, int(idx)])
        )
        r = rng.random()
        dataset_idx = np.searchsorted(self.cumulative_weights, r)
        dataset = self.sub_datasets[dataset_idx]

        # Get item from selected dataset
        dataset_len = len(dataset)
        if dataset_len == 0:
            # Fallback to first dataset
            dataset = self.sub_datasets[0]
            dataset_len = len(dataset)

        local_idx = int(rng.integers(0, dataset_len))
        chunk = np.array(dataset._data[local_idx: local_idx + self.context_length + 1],
                         dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
