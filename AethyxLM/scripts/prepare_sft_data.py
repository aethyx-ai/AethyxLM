"""Normalize, validate, deduplicate, and split instruction conversations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.sft_dataset import validate_messages


ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def iter_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        payload = payload["data"]
    if not isinstance(payload, list):
        payload = [payload]
    yield from payload


def normalize_record(record):
    if isinstance(record.get("messages"), list):
        raw_messages = record["messages"]
    elif isinstance(record.get("conversations"), list):
        raw_messages = record["conversations"]
    elif isinstance(record.get("prompt"), str) and isinstance(record.get("response"), str):
        raw_messages = [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]
    else:
        raise ValueError("record has no supported conversation fields")

    messages = []
    for message in raw_messages:
        raw_role = message.get("role", message.get("from"))
        content = message.get("content", message.get("value"))
        role = ROLE_ALIASES.get(str(raw_role).lower())
        if role is None:
            raise ValueError(f"unsupported role: {raw_role}")
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        messages.append({"role": role, "content": content.strip()})
    validate_messages(messages)
    return {"messages": messages}


def quality_reason(record, min_chars: int, max_chars: int):
    text = "\n".join(message["content"] for message in record["messages"])
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    words = text.lower().split()
    if len(words) >= 20 and len(set(words)) / len(words) < 0.15:
        return "high_repetition"
    return None


def stable_key(record):
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/sft")
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=20000)
    args = parser.parse_args()
    if not 0 < args.validation_ratio < 0.5:
        parser.error("--validation-ratio must be between 0 and 0.5")

    accepted = {}
    rejected = Counter()
    total = 0
    for path in args.inputs:
        for raw in iter_records(path):
            total += 1
            try:
                record = normalize_record(raw)
            except (AttributeError, TypeError, ValueError) as error:
                rejected[f"invalid:{type(error).__name__}"] += 1
                continue
            reason = quality_reason(record, args.min_chars, args.max_chars)
            if reason:
                rejected[reason] += 1
                continue
            key = stable_key(record)
            if key in accepted:
                rejected["duplicate"] += 1
                continue
            accepted[key] = record

    train, validation = [], []
    boundary = int(args.validation_ratio * 10000)
    for key, record in sorted(accepted.items()):
        (validation if int(key[:8], 16) % 10000 < boundary else train).append(record)
    if not train or not validation:
        raise RuntimeError("Prepared split is empty; provide more examples")

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    report = {
        "inputs": [str(path.resolve()) for path in args.inputs],
        "total": total,
        "accepted": len(accepted),
        "train": len(train),
        "validation": len(validation),
        "rejected": dict(rejected),
        "schema": {"messages": [{"role": "user|assistant|system", "content": "text"}]},
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
