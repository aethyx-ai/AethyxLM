"""Evaluate a checkpoint on validation loss and long-context retrieval."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.dataset import AethyxDataset
from evaluation.evaluator import evaluate_language_model
from evaluation.long_context import evaluate_passkey_retrieval
from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--context-lengths", default="32,64,128")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AethyxTokenizer(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected_hash = (
        checkpoint.get("config", {}).get("tokenizer", {}).get("sha256")
        or checkpoint.get("config", {}).get("tokenizer_sha256")
    )
    if expected_hash and expected_hash != tokenizer.sha256:
        raise RuntimeError("checkpoint and evaluation tokenizer fingerprints differ")

    model = GPT.from_checkpoint(args.checkpoint, device=device).eval()
    dataset = AethyxDataset(
        args.validation_data,
        context_length=model.context_length,
        tokenizer_path=args.tokenizer,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    language_metrics = evaluate_language_model(
        model, loader, device=device, max_batches=args.max_batches
    )
    lengths = [int(value) for value in args.context_lengths.split(",") if value]
    lengths = [value for value in lengths if value <= model.context_length]
    long_context = evaluate_passkey_retrieval(
        model, tokenizer, lengths, trials=args.trials
    )
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "tokenizer_sha256": tokenizer.sha256,
        "language_model": asdict(language_metrics),
        "long_context": [
            {**asdict(result), "accuracy": result.accuracy}
            for result in long_context
        ],
    }
    output = json.dumps(payload, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
