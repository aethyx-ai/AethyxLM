"""
AethyxLM
Production BPE Tokenizer Trainer

Author: Aethyx Labs
"""

import json
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
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents

from tokenizer.config import (
    CORPUS_FILE,
    TOKENIZER_FILE,
    VOCAB_SIZE,
    MIN_FREQUENCY,
    SPECIAL_TOKENS,
)


def train_tokenizer():
    """
    Train a Byte-Pair Encoding tokenizer.
    """

    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_FILE}"
        )

    print("=" * 60)
    print("Training AethyxTokenizer...")
    print("=" * 60)

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    tokenizer.normalizer = Sequence([
        NFD(),
        Lowercase(),
        StripAccents(),
    ])

    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train(
        [str(CORPUS_FILE)],
        trainer,
    )

    tokenizer.save(str(TOKENIZER_FILE))

    # Save metadata.json alongside tokenizer.json
    metadata = {
        "vocab_size": VOCAB_SIZE,
        "tokenizer_type": "BPE",
        "special_tokens": SPECIAL_TOKENS,
        "normalizer": {
            "type": "Sequence",
            "components": [
                {"type": "NFD"},
                {"type": "Lowercase"},
                {"type": "StripAccents"}
            ]
        },
        "pre_tokenizer": {
            "type": "ByteLevel"
        },
        "trainer": {
            "type": "BpeTrainer",
            "vocab_size": VOCAB_SIZE,
            "min_frequency": MIN_FREQUENCY,
            "special_tokens": SPECIAL_TOKENS
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_used": str(CORPUS_FILE),
        "corpus_size_bytes": CORPUS_FILE.stat().st_size if CORPUS_FILE.exists() else 0,
    }

    metadata_path = TOKENIZER_FILE.with_name("metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nTraining complete!")
    print(f"Tokenizer saved to:\n{TOKENIZER_FILE}")
    print(f"Metadata saved to:\n{metadata_path}")


if __name__ == "__main__":
    train_tokenizer()