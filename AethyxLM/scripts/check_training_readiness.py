"""Fail-fast audit for the default AethyxLM training run."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(config_path: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_path = ROOT / config["tokenizer"]["tokenizer_file"]
    tokenizer = AethyxTokenizer(tokenizer_path)
    if tokenizer.vocab_size != config["tokenizer"]["vocab_size"]:
        raise ValueError("Tokenizer and configuration vocabulary sizes differ")
    if config["model"]["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("Model and tokenizer vocabulary sizes differ")
    registry_path = ROOT / config["data"]["datasets_file"]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if abs(sum(item["weight"] for item in registry.values()) - 1.0) > 1e-9:
        raise ValueError("Dataset weights must sum to one")

    total_tokens = 0
    for name, item in registry.items():
        for split in ("train", "val"):
            binary = ROOT / item[split]
            sidecar = binary.with_suffix(".bin.meta.json")
            if not binary.is_file() or binary.stat().st_size == 0:
                raise FileNotFoundError(f"Missing binary: {binary}")
            if binary.stat().st_size % np.dtype(np.uint16).itemsize:
                raise ValueError(f"Partial uint16 token in {binary}")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            tokens = binary.stat().st_size // 2
            if metadata["tokens"] != tokens:
                raise ValueError(f"Token count mismatch for {binary}")
            if metadata["tokenizer_sha256"] != tokenizer.sha256:
                raise ValueError(f"Tokenizer mismatch for {binary}")
            if metadata["vocab_size"] != tokenizer.vocab_size:
                raise ValueError(f"Vocabulary mismatch for {binary}")
            total_tokens += tokens

    expected_tokens = config["data"].get("total_tokens")
    if expected_tokens is not None and total_tokens != int(expected_tokens):
        raise ValueError(
            f"Prepared binaries contain {total_tokens:,} tokens, expected "
            f"{int(expected_tokens):,}"
        )

    training = config["training"]
    planned_tokens = training.get("planned_tokens")
    planned_world_size = training.get("planned_world_size")
    if planned_tokens is not None and planned_world_size is not None:
        sequences_per_step = (
            int(config["data"]["batch_size"])
            * int(training["grad_accum_steps"])
            * int(planned_world_size)
        )
        schedule = training.get("context_schedule") or [
            {"step": 0, "context_length": config["model"]["context_length"]}
        ]
        calculated = 0
        for index, stage in enumerate(schedule):
            start = int(stage["step"])
            end = (
                int(schedule[index + 1]["step"])
                if index + 1 < len(schedule)
                else int(training["max_steps"])
            )
            if end > start:
                calculated += (end - start) * sequences_per_step * int(
                    stage["context_length"]
                )
        if calculated != int(planned_tokens):
            raise ValueError(
                f"Context schedule plans {calculated:,} tokens, configured value is "
                f"{int(planned_tokens):,}"
            )

    model = GPT(config=config["model"])
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"READY: {parameters:,} parameters, {tokenizer.vocab_size:,} vocabulary")
    print(f"Datasets: {len(registry)}, binary tokens across splits: {total_tokens:,}")
    print("Start with: python train.py --config configs/train_config_modern.json --device cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_config_modern.json"))
    main(parser.parse_args().config)
