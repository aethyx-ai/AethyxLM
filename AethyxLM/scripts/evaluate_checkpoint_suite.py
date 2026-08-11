"""Evaluate and compare unique AethyxLM checkpoint steps."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.checkpoint_suite import evaluate_checkpoint
from tokenizer.tokenizer import AethyxTokenizer


def checkpoint_candidates(paths, checkpoint_dir):
    candidates = [Path(path) for path in paths]
    if checkpoint_dir:
        candidates.extend(Path(checkpoint_dir).glob("*.pt"))
    resolved = []
    seen = set()
    for path in candidates:
        path = path.expanduser().resolve()
        if path.is_file() and path not in seen:
            resolved.append(path)
            seen.add(path)
    if not resolved:
        raise FileNotFoundError("No checkpoints were supplied or discovered")
    return resolved


def unique_checkpoint_steps(candidates):
    """Resolve aliases before expensive model and dataset evaluation."""
    def preference(path):
        if path.stem.startswith("checkpoint_step_"):
            return 0
        if path.name == "checkpoint_latest.pt":
            return 1
        if path.name == "checkpoint_best.pt":
            return 2
        return 3

    unique = []
    seen_steps = set()
    for path in sorted(candidates, key=lambda item: (preference(item), item.name)):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise RuntimeError(f"Not an AethyxLM training checkpoint: {path}")
        step = int(checkpoint.get("step", -1))
        if step < 0:
            raise RuntimeError(f"Checkpoint does not declare a valid step: {path}")
        if step in seen_steps:
            print(f"Skipping duplicate alias at step {step}: {path}")
            continue
        seen_steps.add(step)
        unique.append(path)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="*", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer/tokenizer.json")
    parser.add_argument("--datasets", type=Path, default=ROOT / "configs/datasets.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--context-lengths", default="128,256,512")
    parser.add_argument("--long-context-trials", type=int, default=2)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results/checkpoints.json")
    args = parser.parse_args()

    tokenizer = AethyxTokenizer(args.tokenizer)
    registry = json.loads(args.datasets.read_text(encoding="utf-8"))
    lengths = [int(value) for value in args.context_lengths.split(",") if value]
    results = []
    candidates = checkpoint_candidates(args.checkpoints, args.checkpoint_dir)
    for checkpoint_path in unique_checkpoint_steps(candidates):
        record = evaluate_checkpoint(
            checkpoint_path,
            tokenizer,
            registry,
            ROOT,
            args.device,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            context_lengths=lengths,
            long_context_trials=args.long_context_trials,
            generation_tokens=args.generation_tokens,
        )
        results.append(record)
        print(f"Evaluated step {record['step']}: {checkpoint_path.name}")

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warning": (
            "Generation repetition metrics are deterministic diagnostics, not a "
            "substitute for human or benchmark-based coherence judgments."
        ),
        "checkpoints": sorted(results, key=lambda item: item["step"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
