"""Build a bounded, reproducible multilingual corpus for AethyxLM tokenizer v2.

The script streams IndicCorpV2 instead of downloading the complete dataset.  It
also samples the existing English training text at evenly spaced byte offsets so
that tokenizer research does not duplicate the full local corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import locale
from pathlib import Path
from typing import BinaryIO

from datasets import load_dataset


INDICCORP_SPLITS = (
    "asm_Beng",
    "ben_Beng",
    "brx_Deva",
    "doi_Deva",
    "gom_Deva",
    "guj_Gujr",
    "hin_Deva",
    "kan_Knda",
    "kas_Arab",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "mni_Mtei",
    "npi_Deva",
    "ory_Orya",
    "pan_Guru",
    "san_Deva",
    "snd_Deva",
    "tam_Taml",
    "tel_Telu",
    "urd_Arab",
    "khasi",
    "santhali",
)


def _write_document(output: BinaryIO, text: str, remaining: int) -> int:
    text = " ".join(text.replace("\x00", " ").split())
    if not text:
        return 0
    payload = ("<DOC>\n" + text + "\n<EOS>\n").encode("utf-8")
    if len(payload) > remaining:
        payload = payload[:remaining]
        # Never end the corpus with an invalid partial UTF-8 code point.
        payload = payload.decode("utf-8", errors="ignore").encode("utf-8")
    output.write(payload)
    return len(payload)


def sample_local_text(source: Path, output: BinaryIO, byte_budget: int) -> dict:
    size = source.stat().st_size
    chunk_count = min(64, max(1, byte_budget // 16_384))
    chunk_budget = max(1, byte_budget // chunk_count)
    written = 0
    documents = 0

    with source.open("rb") as handle:
        for index in range(chunk_count):
            offset = int(index * max(0, size - chunk_budget) / max(1, chunk_count - 1))
            handle.seek(offset)
            if offset:
                handle.readline()  # discard the partial first line
            raw = handle.read(chunk_budget)
            text = raw.decode("utf-8", errors="ignore")
            used = _write_document(output, text, byte_budget - written)
            written += used
            documents += int(used > 0)
            if written >= byte_budget:
                break
    return {"bytes": written, "documents": documents}


def stream_indic_split(split: str, output: BinaryIO, byte_budget: int) -> dict:
    dataset = load_dataset("ai4bharat/IndicCorpV2", split=split, streaming=True)
    written = 0
    documents = 0
    for row in dataset:
        used = _write_document(output, row["text"], byte_budget - written)
        written += used
        documents += int(used > 0)
        if written >= byte_budget:
            break
    return {"bytes": written, "documents": documents}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_corpus(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict] = {}

    with output.open("wb") as handle:
        if args.english_source:
            english_source = Path(args.english_source)
            sources["eng_Latn"] = sample_local_text(
                english_source, handle, args.bytes_per_language
            )
            sources["eng_Latn"]["source"] = str(english_source.resolve())

        for index, split in enumerate(INDICCORP_SPLITS):
            print(f"Streaming {split} ({index + 1}/{len(INDICCORP_SPLITS)})")
            # Buffer one bounded split before committing it, so a remote read
            # failure cannot leave a partial language sample in the corpus.
            split_buffer = io.BytesIO()
            sources[split] = stream_indic_split(
                split, split_buffer, args.bytes_per_language
            )
            handle.write(split_buffer.getvalue())

    metadata = {
        "format_version": 1,
        "purpose": "bounded representative tokenizer-v2 research corpus",
        "output": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "seed": args.seed,
        "bytes_per_language_target": args.bytes_per_language,
        "language_varieties": len(sources),
        "sources": sources,
        "external_dataset": {
            "name": "AI4Bharat/IndicCorpV2",
            "url": "https://huggingface.co/datasets/ai4bharat/IndicCorpV2",
            "license": "cc-by-4.0",
            "mode": "streaming, bounded sample",
        },
        "limitations": [
            "This is a tokenizer pilot corpus, not a statistically complete pretraining corpus.",
            "Each external split is sampled from the start of its stream and is not a uniform sample of the full split.",
        ],
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="tokenizer/data/multilingual_corpus.txt"
    )
    parser.add_argument("--english-source", default="data/train.txt")
    parser.add_argument("--bytes-per-language", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bytes_per_language <= 0:
        parser.error("--bytes-per-language must be positive")
    if not args.english_source:
        args.english_source = None
    elif not Path(args.english_source).is_file():
        parser.error(f"English source does not exist: {args.english_source}")
    if locale.getpreferredencoding(False).lower().replace("-", "") != "utf8":
        print(
            "Note: on Windows, set PYTHONUTF8=1 if the datasets text decoder "
            "does not select UTF-8."
        )
    return args


if __name__ == "__main__":
    result = build_corpus(parse_args())
    print(json.dumps(result, indent=2))
