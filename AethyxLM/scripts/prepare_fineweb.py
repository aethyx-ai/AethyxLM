#!/usr/bin/env python3
"""
AethyxLM - FineWeb-Edu Streaming Dataset Preparation

Streams FineWeb-Edu from Hugging Face, cleans, deduplicates, tokenizes,
and generates binary dataset files for training.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from tokenizer.tokenizer import AethyxTokenizer


class FineWebPreparer:
    """Prepares FineWeb-Edu dataset for AethyxLM training."""

    def __init__(
        self,
        target_gb: float = None,
        target_documents: int = None,
        val_split: float = 0.01,
        output_dir: str = "data",
        tokenizer_path: str = "tokenizer/tokenizer.json",
    ):
        self.target_gb = target_gb
        self.target_documents = target_documents
        self.val_split = val_split
        self.output_dir = Path(output_dir)
        self.tokenizer_path = tokenizer_path

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tokenizer
        self.tokenizer = AethyxTokenizer(tokenizer_path)

        # Statistics
        self.stats = {
            "processed": 0,
            "accepted": 0,
            "duplicates": 0,
            "rejected": 0,
            "tokens": 0,
            "start_time": time.time(),
        }

        # Deduplication
        self.seen_hashes = set()

        # Output files
        self.train_bin = self.output_dir / "fineweb_train.bin"
        self.val_bin = self.output_dir / "fineweb_val.bin"
        self.metadata_path = self.output_dir / "fineweb_metadata.json"

        # Target in bytes
        if target_gb:
            self.target_bytes = int(target_gb * 1024 * 1024 * 1024)
        else:
            self.target_bytes = None

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Unicode normalization
        import unicodedata
        text = unicodedata.normalize("NFKC", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize whitespace
        import re
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip
        text = text.strip()

        return text

    def validate_text(self, text: str) -> bool:
        """Validate text quality."""
        if not text:
            return False

        # Skip if too short
        if len(text) < 100:
            return False

        # Skip if mostly whitespace
        if text.strip() == "":
            return False

        # Check for corrupted unicode
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            return False

        # Check for reasonable character diversity
        unique_chars = len(set(text))
        if unique_chars < 10:
            return False

        return True

    def get_text_hash(self, text: str) -> str:
        """Generate SHA256 hash for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate."""
        text_hash = self.get_text_hash(text)
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False

    def process_document(self, doc: dict) -> list:
        """Process a single document, return list of token IDs."""
        # Get text from document
        text = doc.get("text", "")
        if not text:
            return []

        # Clean
        text = self.clean_text(text)

        # Validate
        if not self.validate_text(text):
            return []

        # Check duplicate
        if self.is_duplicate(text):
            return []

        # Tokenize
        try:
            token_ids = self.tokenizer.encode(text)
        except Exception as e:
            return []

        return token_ids

    def write_tokens(self, token_ids: list, f_out) -> int:
        """Write token IDs to binary file."""
        import numpy as np
        if not token_ids:
            return 0
        arr = np.array(token_ids, dtype=np.uint16)
        arr.tofile(f_out)
        return len(arr)

    def print_progress(self):
        """Print progress statistics."""
        elapsed = time.time() - self.stats["start_time"]
        tokens_per_sec = self.stats["tokens"] / max(elapsed, 1)
        mb_per_sec = (self.stats["tokens"] * 2) / (1024 * 1024 * max(elapsed, 1))

        print(
            f"\rProcessed: {self.stats['processed']:,} | "
            f"Accepted: {self.stats['accepted']:,} | "
            f"Duplicates: {self.stats['duplicates']:,} | "
            f"Rejected: {self.stats['rejected']:,} | "
            f"Tokens: {self.stats['tokens']:,} | "
            f"{tokens_per_sec:,.0f} tok/s | "
            f"{mb_per_sec:.2f} MB/s",
            end="",
            flush=True,
        )

    def run(self):
        """Main processing loop."""
        print("=" * 60)
        print("FineWeb-Edu Preparation")
        print("=" * 60)
        print(f"Target: {self.target_gb} GB" if self.target_gb else f"Target: {self.target_documents:,} docs")
        print(f"Val split: {self.val_split * 100:.1f}%")
        print(f"Output: {self.output_dir}")
        print(f"Tokenizer: {self.tokenizer_path}")
        print("=" * 60)

        # Load dataset in streaming mode
        print("Loading FineWeb-Edu (streaming)...")
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

        # Open output files
        with open(self.train_bin, "wb") as f_train, open(self.val_bin, "wb") as f_val:
            for doc in dataset:
                self.stats["processed"] += 1

                # Process document
                token_ids = self.process_document(doc)

                if not token_ids:
                    self.stats["rejected"] += 1
                    continue

                # Accepted
                self.stats["accepted"] += 1
                self.stats["tokens"] += len(token_ids)

                # Write to train or val
                if self.stats["accepted"] % int(1 / self.val_split) == 0:
                    written = self.write_tokens(token_ids, self.val_bin)
                else:
                    written = self.write_tokens(token_ids, self.train_bin)

                # Check target
                if self.target_bytes:
                    current_size = self.train_bin.stat().st_size + self.val_bin.stat().st_size
                    if current_size >= self.target_bytes:
                        print(f"\nTarget size reached: {current_size / (1024**3):.2f} GB")
                        break

                if self.target_documents and self.stats["accepted"] >= self.target_documents:
                    print(f"\nTarget documents reached: {self.stats['accepted']:,}")
                    break

                # Progress
                if self.stats["processed"] % 100 == 0:
                    self.print_progress()

        # Final stats
        print()
        self.finalize()

    def finalize(self):
        """Finalize and generate metadata."""
        elapsed = time.time() - self.stats["start_time"]
        train_size = self.train_bin.stat().st_size if self.train_bin.exists() else 0
        val_size = self.val_bin.stat().st_size if self.val_bin.exists() else 0
        total_size = train_size + val_size

        # Count tokens in train/val
        import numpy as np
        train_tokens = 0
        val_tokens = 0
        if self.train_bin.exists():
            train_tokens = len(np.memmap(self.train_bin, dtype=np.uint16, mode='r'))
        if self.val_bin.exists():
            val_tokens = len(np.memmap(self.val_bin, dtype=np.uint16, mode='r'))

        # Calculate average document length
        avg_length = 0
        if self.stats["accepted"] > 0:
            avg_length = self.stats["tokens"] / self.stats["accepted"]

        metadata = {
            "dataset": "FineWeb-Edu (sample-10BT)",
            "date": datetime.now().isoformat(),
            "documents_processed": self.stats["processed"],
            "documents_accepted": self.stats["accepted"],
            "documents_duplicates": self.stats["duplicates"],
            "documents_rejected": self.stats["rejected"],
            "total_tokens": self.stats["tokens"],
            "average_document_length": avg_length,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "train_size_bytes": train_size,
            "val_size_bytes": val_size,
            "total_size_bytes": total_size,
            "train_size_gb": train_size / (1024**3),
            "val_size_gb": val_size / (1024**3),
            "total_size_gb": total_size / (1024**3),
            "elapsed_seconds": elapsed,
            "tokenizer": str(self.tokenizer_path),
            "val_split": self.val_split,
        }

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Print summary
        print("=" * 60)
        print("FineWeb-Edu Preparation Complete")
        print("=" * 60)
        print(f"Documents processed: {self.stats['processed']:,}")
        print(f"Documents accepted: {self.stats['accepted']:,}")
        print(f"Duplicates removed: {self.stats['duplicates']:,}")
        print(f"Rejected: {self.stats['rejected']:,}")
        print(f"Tokens: {self.stats['tokens']:,}")
        print(f"Train tokens: {train_tokens:,}")
        print(f"Val tokens: {val_tokens:,}")
        print(f"Train size: {train_size / (1024**3):.2f} GB")
        print(f"Val size: {val_size / (1024**3):.2f} GB")
        print(f"Total size: {total_size / (1024**3):.2f} GB")
        print(f"Time: {time.time() - self.stats['start_time']:.1f}s")
        print(f"Metadata saved to: {self.metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FineWeb-Edu dataset for AethyxLM training"
    )
    parser.add_argument(
        "--target-gb",
        type=float,
        default=None,
        help="Target size in GB (e.g., 10 for 10GB)",
    )
    parser.add_argument(
        "--target-documents",
        type=int,
        default=None,
        help="Target number of documents",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.01,
        help="Validation split ratio (default: 0.01)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer/tokenizer.json",
        help="Tokenizer path (default: tokenizer/tokenizer.json)",
    )
    args = parser.parse_args()

    if not args.target_gb and not args.target_documents:
        parser.error("Either --target-gb or --target-documents must be specified")

    preparer = FineWebPreparer(
        target_gb=args.target_gb,
        target_documents=args.target_documents,
        val_split=args.val_split,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
    )
    preparer.run()


if __name__ == "__main__":
    main()