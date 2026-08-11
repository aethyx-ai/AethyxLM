"""Summarize append-only AethyxLM JSONL experiment logs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = defaultdict(list)
    with args.metrics.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}") from error
            runs[record.get("run_id", "unknown")].append(record)

    summary = {"source": str(args.metrics.resolve()), "runs": {}}
    for run_id, records in runs.items():
        training = [item for item in records if item.get("event") == "train_metrics"]
        validation = [item for item in records if item.get("event") == "validation"]
        summary["runs"][run_id] = {
            "events": len(records),
            "last_step": max((item.get("step", 0) for item in records), default=0),
            "last_train": training[-1] if training else None,
            "best_validation": min(
                validation, key=lambda item: item.get("loss", float("inf")), default=None
            ),
        }
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
