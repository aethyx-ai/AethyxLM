"""Stream a bounded, reproducible instruction-tuning bundle from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_sft_data import normalize_record, quality_reason, stable_key


def resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _stream_source(source: dict, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The 'datasets' package is required. Run: pip install -r requirements.txt"
        ) from error

    name = source["dataset"]
    subset = source.get("subset")
    kwargs = {
        "split": source.get("split", "train"),
        "streaming": True,
    }
    if source.get("revision"):
        kwargs["revision"] = source["revision"]
    dataset = load_dataset(name, subset, **kwargs) if subset else load_dataset(name, **kwargs)
    buffer_size = int(source.get("shuffle_buffer", 10_000))
    if buffer_size > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    return dataset


def prepare_from_config(config_path: str | Path, force: bool = False) -> dict:
    """Prepare train/validation JSONL files described by an SFT config."""
    config_path = resolve(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    prep = config.get("preparation")
    if not prep:
        raise RuntimeError(f"{config_path} has no 'preparation' section")

    train_path = resolve(data_config["train"])
    validation_path = resolve(data_config["validation"])
    report_path = train_path.parent / "report.json"
    if train_path.is_file() and validation_path.is_file() and not force:
        return {
            "status": "existing",
            "train_path": str(train_path),
            "validation_path": str(validation_path),
        }

    ratio = float(prep.get("validation_ratio", 0.02))
    if not 0 < ratio < 0.5:
        raise ValueError("preparation.validation_ratio must be between 0 and 0.5")
    min_chars = int(prep.get("min_chars", 20))
    max_chars = int(prep.get("max_chars", 8_000))
    seed = int(config.get("seed", 42))
    boundary = int(ratio * 10_000)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_partial = train_path.with_suffix(train_path.suffix + ".partial")
    validation_partial = validation_path.with_suffix(validation_path.suffix + ".partial")
    for partial in (train_partial, validation_partial):
        partial.unlink(missing_ok=True)

    seen: set[str] = set()
    rejected = Counter()
    source_reports = []
    train_count = validation_count = 0

    try:
        with train_partial.open("w", encoding="utf-8", newline="\n") as train_handle, \
             validation_partial.open("w", encoding="utf-8", newline="\n") as val_handle:
            for source_index, source in enumerate(prep["sources"]):
                limit = int(source["limit"])
                max_scanned = int(source.get("max_scanned", max(limit * 10, limit)))
                accepted = scanned = 0
                print(
                    f"[DATA] Streaming {source['dataset']}"
                    f"{('/' + source['subset']) if source.get('subset') else ''} "
                    f"(target={limit:,})"
                )
                for raw in _stream_source(source, seed + source_index):
                    if accepted >= limit or scanned >= max_scanned:
                        break
                    scanned += 1
                    try:
                        record = normalize_record(raw)
                    except (AttributeError, TypeError, ValueError) as error:
                        rejected[f"invalid:{type(error).__name__}"] += 1
                        continue
                    reason = quality_reason(record, min_chars, max_chars)
                    if reason:
                        rejected[reason] += 1
                        continue
                    key = stable_key(record)
                    if key in seen:
                        rejected["duplicate"] += 1
                        continue
                    seen.add(key)
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    if int(key[:8], 16) % 10_000 < boundary:
                        val_handle.write(line)
                        validation_count += 1
                    else:
                        train_handle.write(line)
                        train_count += 1
                    accepted += 1
                if accepted < limit:
                    raise RuntimeError(
                        f"{source['dataset']} yielded only {accepted:,} usable examples "
                        f"after scanning {scanned:,}; requested {limit:,}"
                    )
                source_reports.append(
                    {
                        "dataset": source["dataset"],
                        "subset": source.get("subset"),
                        "accepted": accepted,
                        "scanned": scanned,
                        "license": source.get("license", "see dataset card"),
                    }
                )
                print(f"[DATA] Accepted {accepted:,} examples")

        if train_count == 0 or validation_count == 0:
            raise RuntimeError("Prepared split is empty; increase the bundle size")
        os.replace(train_partial, train_path)
        os.replace(validation_partial, validation_path)
    except BaseException:
        train_partial.unlink(missing_ok=True)
        validation_partial.unlink(missing_ok=True)
        raise

    report = {
        "status": "prepared",
        "seed": seed,
        "train": train_count,
        "validation": validation_count,
        "total": train_count + validation_count,
        "sources": source_reports,
        "rejected": dict(rejected),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_rtx3050_6gb.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_from_config(args.config, force=args.force)


if __name__ == "__main__":
    main()
