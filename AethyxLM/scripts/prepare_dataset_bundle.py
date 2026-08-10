"""Prepare a manifest-defined collection without retaining raw source text."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_fineweb import FineWebPreparer
from scripts.prepare_fineweb import atomic_write_json
from tokenizer.tokenizer import AethyxTokenizer


def ensure_token_sidecars(source: dict, output_dir: Path, tokenizer_path: str):
    tokenizer = AethyxTokenizer(tokenizer_path)
    digest = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
    metadata = json.loads(
        (output_dir / f"{source['name']}_metadata.json").read_text(encoding="utf-8")
    )
    metadata["source_revision"] = source.get("source_revision")
    metadata["tokenizer_sha256"] = digest
    atomic_write_json(output_dir / f"{source['name']}_metadata.json", metadata)
    for binary_split, suffix, count_key in (
        ("train", "train", "train_tokens"),
        ("validation", "val", "validation_tokens"),
    ):
        path = output_dir / f"{source['name']}_{suffix}.bin"
        atomic_write_json(
            path.with_suffix(".bin.meta.json"),
            {
                "tokenizer_file": str(tokenizer.path),
                "tokenizer_sha256": digest,
                "vocab_size": tokenizer.vocab_size,
                "tokens": metadata[count_key],
                "dtype": "uint16",
                "dataset": source["dataset_name"],
                "dataset_config": source.get("dataset_config"),
                "source_split": source.get("source_split", "train"),
                "source_revision": source.get("source_revision"),
                "binary_split": binary_split,
            },
        )


def completed_output(source: dict, output_dir: Path) -> bool:
    prefix = source["name"]
    metadata_path = output_dir / f"{prefix}_metadata.json"
    train_path = output_dir / f"{prefix}_train.bin"
    val_path = output_dir / f"{prefix}_val.bin"
    if not metadata_path.exists() or not train_path.exists() or not val_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        metadata.get("status") == "complete"
        and metadata.get("train_size_bytes") == train_path.stat().st_size
        and metadata.get("validation_size_bytes") == val_path.stat().st_size
        and metadata.get("total_size_bytes")
        == train_path.stat().st_size + val_path.stat().st_size
    )


def prepare_bundle(args):
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.bundle not in manifest:
        raise KeyError(f"Unknown bundle {args.bundle!r}")
    bundle = manifest[args.bundle]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_bytes = sum(int(source["target_gb"] * 1024**3) for source in bundle["sources"])
    reserve_bytes = int(bundle.get("reserve_gb", 2.0) * 1024**3)
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < target_bytes + reserve_bytes:
        raise OSError(
            f"Bundle needs {target_bytes / 1024**3:.2f} GiB plus "
            f"{reserve_bytes / 1024**3:.2f} GiB reserve, but only "
            f"{free_bytes / 1024**3:.2f} GiB is free"
        )

    print(
        f"Preparing {args.bundle}: {len(bundle['sources'])} sources, "
        f"maximum binary output {target_bytes / 1024**3:.2f} GiB"
    )
    for source in bundle["sources"]:
        if completed_output(source, output_dir):
            ensure_token_sidecars(source, output_dir, bundle["tokenizer"])
            print(f"[skip] Verified complete: {source['name']}")
            continue
        state_path = output_dir / f"{source['name']}_state.json"
        resume = state_path.exists()
        print(f"[{'resume' if resume else 'start'}] {source['name']}")
        preparer = FineWebPreparer(
            target_gb=source["target_gb"],
            target_documents=None,
            val_split=source.get("val_split", 0.01),
            output_dir=str(output_dir),
            tokenizer_path=bundle["tokenizer"],
            resume=resume,
            resume_from=None,
            buffer_tokens=args.buffer_tokens,
            progress_seconds=args.progress_seconds,
            dedup_window=source.get("dedup_window", 0),
            dataset_name=source["dataset_name"],
            dataset_config=source.get("dataset_config"),
            source_split=source.get("source_split", "train"),
            text_field=source.get("text_field", "text"),
            output_prefix=source["name"],
            source_revision=source.get("source_revision"),
        )
        status = preparer.run()
        if status != "complete":
            raise RuntimeError(f"{source['name']} ended with status {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/storage_bounded_sources.json")
    parser.add_argument("--bundle", default="multilingual_v2_32k")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--buffer-tokens", type=int, default=500000)
    parser.add_argument("--progress-seconds", type=float, default=10.0)
    prepare_bundle(parser.parse_args())
