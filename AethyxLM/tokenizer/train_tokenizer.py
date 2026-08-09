"""
AethyxLM
Production BPE Tokenizer Trainer

Author: Aethyx Labs
"""

import json
import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for package imports (needed by subprocesses)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, NFD, NFKC, Lowercase, StripAccents
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from tokenizer.config import (
    CORPUS_FILE,
    TOKENIZER_FILE,
    VOCAB_SIZE,
    MIN_FREQUENCY,
    SPECIAL_TOKENS,
    STRUCTURAL_TOKENS,
    TOKENIZER_V2_FILE,
)


def train_tokenizer(
    corpus_file=CORPUS_FILE,
    output_file=TOKENIZER_V2_FILE,
    vocab_size=VOCAB_SIZE,
    legacy_normalization=False,
):
    """
    Train a Byte-Pair Encoding tokenizer.
    """

    corpus_file = Path(corpus_file)
    output_file = Path(output_file)
    if not corpus_file.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_file}"
        )

    print("=" * 60)
    print("Training AethyxTokenizer...")
    print("=" * 60)

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    tokenizer.normalizer = (
        Sequence([NFD(), Lowercase(), StripAccents()])
        if legacy_normalization
        else NFKC()
    )

    # Do not silently inject a leading space: exact normalized round-tripping is
    # important for code, structured context, and Indic scripts.
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS + STRUCTURAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train(
        [str(corpus_file)],
        trainer,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_file))

    # Save metadata.json alongside tokenizer.json
    metadata = {
        "vocab_size": tokenizer.get_vocab_size(),
        "requested_vocab_size": vocab_size,
        "format_version": 2,
        "tokenizer_type": "BPE",
        "special_tokens": SPECIAL_TOKENS + STRUCTURAL_TOKENS,
        "normalizer": (
            {"type": "Sequence", "legacy_lowercase_and_strip_accents": True}
            if legacy_normalization
            else {"type": "NFKC", "preserves_case_and_accents": True}
        ),
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
        },
        "decoder": {
            "type": "ByteLevel"
        },
        "trainer": {
            "type": "BpeTrainer",
            "vocab_size": vocab_size,
            "min_frequency": MIN_FREQUENCY,
            "special_tokens": SPECIAL_TOKENS + STRUCTURAL_TOKENS
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_used": str(corpus_file),
        "corpus_size_bytes": corpus_file.stat().st_size,
    }

    metadata_path = output_file.with_name(output_file.stem + "_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nTraining complete!")
    print(f"Tokenizer saved to:\n{output_file}")
    print(f"Metadata saved to:\n{metadata_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the AethyxLM v2 tokenizer")
    parser.add_argument("--corpus", default=str(CORPUS_FILE))
    parser.add_argument("--output", default=str(TOKENIZER_V2_FILE))
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--legacy-normalization", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(args.corpus, args.output, args.vocab_size, args.legacy_normalization)
