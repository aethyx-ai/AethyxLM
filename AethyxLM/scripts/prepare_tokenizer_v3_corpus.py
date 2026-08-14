"""Build stratified train/held-out text for AethyxLM tokenizer v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_fineweb import atomic_write_json
from scripts.source_filters import (
    is_high_quality_code,
    matches_required_values,
    normalize_text,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_source(source: dict, seed: int, shuffle_buffer: int):
    args = [source["dataset_name"]]
    if source.get("dataset_config") is not None:
        args.append(source["dataset_config"])
    fields = [source.get("text_field", "text"), *source.get("required_values", {})]
    if source.get("code_quality_filter"):
        fields.append("path")
    fields = list(dict.fromkeys(fields))
    common = {
        "split": source.get("source_split", "train"),
        "streaming": True,
        "revision": source.get("source_revision"),
    }
    try:
        dataset = load_dataset(*args, **common, columns=fields, batch_size=100)
    except ValueError as error:
        if "doesn't have a 'columns' key" not in str(error):
            raise
        dataset = load_dataset(*args, **common)
    # Avoid training the tokenizer only on the first repository/domain in a shard.
    return dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)


def encode_document(text: str, max_bytes: int) -> bytes:
    prefix = b"<DOC>\n"
    suffix = b"\n<EOS>\n"
    content_budget = max(1, max_bytes - len(prefix) - len(suffix))
    content = text.encode("utf-8")[:content_budget]
    content = content.decode("utf-8", errors="ignore").encode("utf-8")
    return prefix + content + suffix


def local_documents(path: Path):
    text = path.read_text(encoding="utf-8")
    for part in text.split("<DOC>\n")[1:]:
        document = part.rsplit("\n<EOS>", 1)[0]
        if document:
            yield document


def build(args: argparse.Namespace) -> dict:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.bundle not in manifest:
        raise KeyError(f"Unknown bundle {args.bundle!r}")
    bundle = manifest[args.bundle]
    output = Path(args.output)
    heldout = Path(args.heldout)
    output.parent.mkdir(parents=True, exist_ok=True)
    heldout.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_suffix(output.suffix + ".tmp")
    heldout_tmp = heldout.with_suffix(heldout.suffix + ".tmp")
    records = []

    with output_tmp.open("wb") as train_handle, heldout_tmp.open("wb") as heldout_handle:
        if args.local_corpus:
            local_path = Path(args.local_corpus)
            local_train_bytes = local_heldout_bytes = 0
            local_train_documents = local_heldout_documents = 0
            for text in local_documents(local_path):
                digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
                payload = encode_document(text, args.max_document_bytes)
                if int.from_bytes(digest, "big") % 10 == 0:
                    heldout_handle.write(payload)
                    local_heldout_bytes += len(payload)
                    local_heldout_documents += 1
                else:
                    for _ in range(args.local_corpus_repeats):
                        train_handle.write(payload)
                        local_train_bytes += len(payload)
                    local_train_documents += 1
            records.append(
                {
                    "name": "local_multilingual_indic_v2",
                    "dataset_name": "local",
                    "dataset_config": None,
                    "source_split": "train",
                    "source_revision": sha256(local_path),
                    "train_bytes": local_train_bytes,
                    "heldout_bytes": local_heldout_bytes,
                    "documents": local_train_documents + local_heldout_documents,
                    "train_documents": local_train_documents,
                    "heldout_documents": local_heldout_documents,
                    "rejected": 0,
                }
            )
            print(
                f"[local] {local_path}: {local_train_bytes / 1e6:.1f} MB train + "
                f"{local_heldout_bytes / 1e6:.1f} MB held-out",
                flush=True,
            )
        for index, source in enumerate(bundle["sources"]):
            train_budget = int(source["tokenizer_sample_bytes"])
            if train_budget <= 0:
                print(f"[{index + 1}/{len(bundle['sources'])}] {source['name']}: skipped for tokenizer corpus")
                records.append(
                    {
                        "name": source["name"],
                        "dataset_name": source["dataset_name"],
                        "dataset_config": source.get("dataset_config"),
                        "source_split": source.get("source_split", "train"),
                        "source_revision": source.get("source_revision"),
                        "train_bytes": 0,
                        "heldout_bytes": 0,
                        "documents": 0,
                        "train_documents": 0,
                        "heldout_documents": 0,
                        "rejected": 0,
                    }
                )
                continue
            heldout_budget = max(64_000, train_budget // 10)
            train_bytes = heldout_bytes = documents = rejected = 0
            train_documents = heldout_documents = 0
            seen = set()
            print(
                f"[{index + 1}/{len(bundle['sources'])}] {source['name']}: "
                f"{train_budget / 1e6:.1f} MB train + {heldout_budget / 1e6:.1f} MB held-out",
                flush=True,
            )
            for row in stream_source(source, args.seed + index, args.shuffle_buffer):
                if not matches_required_values(row, source.get("required_values", {})):
                    rejected += 1
                    continue
                raw = row.get(source.get("text_field", "text"))
                if not isinstance(raw, str):
                    rejected += 1
                    continue
                text = normalize_text(raw, source.get("preserve_formatting", False))
                if len(text) < 100 or len(set(text)) < 10:
                    rejected += 1
                    continue
                if source.get("code_quality_filter") and not is_high_quality_code(row, text):
                    rejected += 1
                    continue
                digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
                if digest in seen:
                    rejected += 1
                    continue
                seen.add(digest)
                payload = encode_document(text, args.max_document_bytes)
                route_to_heldout = int.from_bytes(digest, "big") % 10 == 0
                if route_to_heldout and heldout_bytes < heldout_budget:
                    remaining = heldout_budget - heldout_bytes
                    payload = payload[:remaining].decode("utf-8", errors="ignore").encode("utf-8")
                    heldout_handle.write(payload)
                    heldout_bytes += len(payload)
                    heldout_documents += 1
                elif train_bytes < train_budget:
                    remaining = train_budget - train_bytes
                    payload = payload[:remaining].decode("utf-8", errors="ignore").encode("utf-8")
                    train_handle.write(payload)
                    train_bytes += len(payload)
                    train_documents += 1
                documents += 1
                if train_bytes >= train_budget and heldout_bytes >= heldout_budget:
                    break
            else:
                raise RuntimeError(
                    f"{source['name']} exhausted before its tokenizer sample was filled"
                )
            records.append(
                {
                    "name": source["name"],
                    "dataset_name": source["dataset_name"],
                    "dataset_config": source.get("dataset_config"),
                    "source_split": source.get("source_split", "train"),
                    "source_revision": source.get("source_revision"),
                    "train_bytes": train_bytes,
                    "heldout_bytes": heldout_bytes,
                    "documents": documents,
                    "train_documents": train_documents,
                    "heldout_documents": heldout_documents,
                    "rejected": rejected,
                }
            )
            train_handle.flush()
            heldout_handle.flush()
            os.fsync(train_handle.fileno())
            os.fsync(heldout_handle.fileno())

    os.replace(output_tmp, output)
    os.replace(heldout_tmp, heldout)
    metadata = {
        "format_version": 3,
        "purpose": "stratified AethyxLM tokenizer-v3 training and held-out evaluation",
        "manifest": str(Path(args.manifest).resolve()),
        "bundle": args.bundle,
        "seed": args.seed,
        "train": {"path": str(output.resolve()), "bytes": output.stat().st_size, "sha256": sha256(output)},
        "heldout": {"path": str(heldout.resolve()), "bytes": heldout.stat().st_size, "sha256": sha256(heldout)},
        "sources": records,
    }
    atomic_write_json(output.with_suffix(output.suffix + ".metadata.json"), metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/pretrain_8b_sources.json")
    parser.add_argument("--bundle", default="aethyxlm_v3_48k_8b")
    parser.add_argument("--output", default="tokenizer/data/tokenizer_v3_corpus.txt")
    parser.add_argument("--heldout", default="tokenizer/data/tokenizer_v3_heldout.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--max-document-bytes", type=int, default=65536)
    parser.add_argument(
        "--local-corpus", default="tokenizer/data/multilingual_corpus.txt"
    )
    parser.add_argument(
        "--local-corpus-repeats",
        type=int,
        default=5,
        help="Oversample Indic tokenizer-training documents; held-out documents are never repeated.",
    )
    build(parser.parse_args())
