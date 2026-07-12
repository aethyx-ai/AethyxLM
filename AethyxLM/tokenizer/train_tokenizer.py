"""
AethyxLM
Production BPE Tokenizer Trainer

Author: Aethyx Labs
"""

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents

from .config import (
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

    print("\nTraining complete!")
    print(f"Tokenizer saved to:\n{TOKENIZER_FILE}")


if __name__ == "__main__":
    train_tokenizer()