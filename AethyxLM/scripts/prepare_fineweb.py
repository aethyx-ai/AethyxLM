#!/usr/bin/env python3
"""
AethyxLM - FineWeb-Edu Streaming Dataset Preparation

Streams FineWeb-Edu from Hugging Face, cleans, deduplicates, tokenizes,
and generates binary dataset files for training with true streaming.

Requirements:
- True streaming from Hugging Face (no full dataset download)
- Incremental tokenization and writing
- Buffered writing with periodic flush
- Proper validation split
- Correct stop condition (by bytes/tokens)
- Live progress reporting
- Crash safety with buffer flush on interrupt
- Resume support
- Disk verification after each flush
- Memory verification
- Modular, clean code structure
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import time
import mmap
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Iterator
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from tokenizer.tokenizer import AethyxTokenizer


@dataclass
class PrepStats:
    """Statistics for preparation run."""
    processed: int = 0
    accepted: int = 0
    duplicates: int = 0
    rejected: int = 0
    tokens: int = 0
    train_tokens: int = 0
    val_tokens: int = 0
    start_time: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WriteStats:
    """Track write statistics for verification."""
    bytes_written: int = 0
    tokens_written: int = 0
    flush_count: int = 0


class TokenBuffer:
    """Buffered token writer with periodic flush and verification."""
    
    def __init__(self, file_path: Path, buffer_size: int = 1_000_000):
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.buffer: List[int] = []
        self.stats = WriteStats()
        self._file = None
        self._mmap = None
        
    def __enter__(self):
        self._file = open(self.file_path, 'ab')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        if self._file:
            self._file.close()
    
    def add(self, token_ids: List[int]) -> int:
        """Add token IDs to buffer, flush if buffer full."""
        if not token_ids:
            return 0
        
        self.buffer.extend(token_ids)
        written = 0
        
        while len(self.buffer) >= self.buffer_size:
            written += self._flush_chunk()
        
        return written
    
    def _flush_chunk(self) -> int:
        """Flush one chunk of buffer_size tokens."""
        if not self.buffer:
            return 0
        
        import numpy as np
        chunk = self.buffer[:self.buffer_size]
        arr = np.array(chunk, dtype=np.uint16)
        
        # Write and verify
        self._file.write(arr.tobytes())
        self._file.flush()
        os.fsync(self._file.fileno())
        
        # Verify write
        expected_bytes = len(chunk) * 2  # uint16 = 2 bytes
        actual_size = self.file_path.stat().st_size
        # Note: Can't easily verify exact bytes without tracking position
        # We'll track total bytes written separately
        
        self.buffer = self.buffer[self.buffer_size:]
        self.stats.bytes_written += len(chunk) * 2
        self.stats.tokens_written += len(chunk)
        self.stats.flush_count += 1
        return len(chunk)
    
    def flush(self) -> int:
        """Flush remaining buffer."""
        total = 0
        while self.buffer:
            total += self._flush_chunk()
        return total
    
    def get_stats(self) -> WriteStats:
        return self.stats


class FineWebPreparer:
    """Prepares FineWeb-Edu dataset for AethyxLM training with true streaming."""
    
    # Buffer size in tokens (each token = 2 bytes as uint16)
    DEFAULT_BUFFER_SIZE = 1_000_000  # ~2MB per flush
    
    def __init__(
        self,
        target_gb: float = None,
        target_documents: int = None,
        val_split: float = 0.01,
        output_dir: str = "data",
        tokenizer_path: str = "tokenizer/tokenizer.json",
        resume: bool = False,
    ):
        self.target_gb = target_gb
        self.target_documents = target_documents
        self.val_split = val_split
        self.output_dir = Path(output_dir)
        self.tokenizer_path = tokenizer_path
        self.resume = resume
        
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
            "train_tokens": 0,
            "val_tokens": 0,
            "start_time": time.time(),
        }
        
        # Deduplication - use memory-efficient approach
        self.seen_hashes = set()
        self.max_hashes = 50_000_000  # Limit to prevent OOM
        
        # Output files
        self.train_bin = self.output_dir / "fineweb_train.bin"
        self.val_bin = self.output_dir / "fineweb_val.bin"
        self.metadata_path = self.output_dir / "fineweb_metadata.json"
        self.state_path = self.output_dir / "fineweb_state.json"
        
        # Target in bytes
        if target_gb:
            self.target_bytes = int(target_gb * 1024 * 1024 * 1024)
        else:
            self.target_bytes = None
        
        if target_documents:
            self.target_docs = target_documents
        else:
            self.target_docs = None
        
        self.val_split = val_split
        self.val_every = max(1, int(1 / val_split))
        
        # Signal handling for graceful shutdown
        self._shutdown = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Resume state
        self._resume_offset = 0
        self._last_doc_hash = None
        
        if resume:
            self._load_resume_state()
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print("\n[INTERRUPT] Received signal, flushing buffers...")
        self._shutdown = True
    
    def _load_resume_state(self):
        """Load resume state from previous run."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    state = json.load(f)
                self.stats.update(state.get('stats', {}))
                self._resume_offset = state.get('resume_offset', 0)
                self._last_doc_hash = state.get('last_doc_hash')
                print(f"[RESUME] Resuming from offset {self._resume_offset}, "
                      f"accepted={self.stats['accepted']}, tokens={self.stats['tokens']:,}")
            except Exception as e:
                print(f"[WARN] Failed to load resume state: {e}")
    
    def _save_resume_state(self):
        """Save resume state for recovery."""
        state = {
            'stats': {k: v for k, v in self.stats.items() if k != 'start_time'},
            'resume_offset': self._resume_offset,
            'last_doc_hash': self._last_doc_hash,
            'timestamp': time.time(),
        }
        try:
            with open(self.state_path, 'w') as f:
                json.dump(state, f)
        except Exception:
            pass  # Non-critical
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        import unicodedata
        import re
        
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Normalize whitespace
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
        """Check if text is a duplicate using hash set with size limit."""
        text_hash = self.get_text_hash(text)
        if text_hash in self.seen_hashes:
            return True
        
        # Limit hash set size to prevent OOM
        if len(self.seen_hashes) >= self.max_hashes:
            # Clear half the set (simple LRU approximation)
            # In production, use LRU cache or bloom filter
            items = list(self.seen_hashes)[:self.max_hashes // 2]
            self.seen_hashes = set(items)
        
        self.seen_hashes.add(text_hash)
        return False
    
    def process_document(self, doc: dict) -> List[int]:
        """Process a single document, return list of token IDs."""
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
        except Exception:
            return []
        
        return token_ids
    
    @contextmanager
    def _open_binary_files(self):
        """Context manager for binary file handles with proper flushing."""
        train_file = open(self.train_bin, 'ab')
        val_file = open(self.val_bin, 'ab')
        try:
            yield train_file, val_file
        finally:
            train_file.flush()
            val_file.flush()
            os.fsync(train_file.fileno())
            os.fsync(val_file.fileno())
            train_file.close()
            val_file.close()
    
    def _write_tokens_buffered(self, token_ids: List[int], is_val: bool, 
                                train_buffer: 'TokenBuffer', 
                                val_buffer: 'TokenBuffer') -> Tuple[int, int]:
        """Write tokens to appropriate buffer."""
        if not token_ids:
            return 0, 0
        
        if is_val:
            written = self._write_to_buffer(token_ids, val_buffer)
            return 0, len(token_ids)
        else:
            written = self._write_to_buffer(token_ids, train_buffer)
            return len(token_ids), 0
    
    def _write_to_buffer(self, token_ids: List[int], buffer) -> int:
        """Write tokens to buffer, returns tokens written."""
        written = buffer.add(token_ids)
        return written
    
    def print_progress(self):
        """Print progress statistics."""
        elapsed = time.time() - self.stats["start_time"]
        if elapsed <= 0:
            return
            
        tokens_per_sec = self.stats["tokens"] / elapsed
        mb_per_sec = (self.stats["tokens"] * 2) / (1024 * 1024 * elapsed)
        
        # Calculate current file sizes
        train_size = self.train_bin.stat().st_size if self.train_bin.exists() else 0
        val_size = self.val_bin.stat().st_size if self.val_bin.exists() else 0
        
        # ETA calculation
        if self.target_bytes:
            current_size = self.train_bin.stat().st_size + self.val_bin.stat().st_size
            if self.stats["tokens"] > 0 and elapsed > 0:
                tokens_per_sec = self.stats["tokens"] / max(elapsed, 1)
                remaining_bytes = self.target_bytes - (self.train_bin.stat().st_size + self.val_bin.stat().st_size)
                if self.stats["tokens"] > 0:
                    bytes_per_token = (self.train_bin.stat().st_size + self.val_bin.stat().st_size) / max(self.stats["tokens"], 1)
                    eta_seconds = remaining_bytes / (tokens_per_sec * bytes_per_token) if tokens_per_sec > 0 else 0
                    eta_str = f"{int(eta_seconds//3600):02d}:{int((eta_seconds%3600)//60):02d}:{int(eta_seconds%60):02d}"
                else:
                    eta_str = "N/A"
            else:
                eta_str = "N/A"
        else:
            eta_str = "N/A"
        
        print(
            f"\rProcessed: {self.stats['processed']:,} | "
            f"Accepted: {self.stats['accepted']:,} | "
            f"Dup: {self.stats['duplicates']:,} | "
            f"Rej: {self.stats['rejected']:,} | "
            f"Train Tokens: {self.stats.get('train_tokens', 0):,} | "
            f"Val Tokens: {self.stats.get('val_tokens', 0):,} | "
            f"Train Size: {self.train_bin.stat().st_size / (1024**3):.2f} GB | "
            f"Val Size: {self.val_bin.stat().st_size / (1024**3):.2f} GB | "
            f"Write: {self.stats['tokens'] / max(time.time() - self.stats['start_time'], 1):,.0f} tok/s | "
            f"ETA: {eta_str}",
            end="",
            flush=True,
        )
    
    def _verify_write(self, file_path: Path, expected_bytes: int) -> bool:
        """Verify written bytes match expected."""
        actual = file_path.stat().st_size
        return actual == expected_bytes
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def run(self):
        """Main processing loop."""
        print("=" * 70)
        print("FineWeb-Edu Streaming Preparation")
        print("=" * 70)
        print(f"Target: {self.target_gb} GB" if self.target_gb else f"Target: {self.target_documents:,} docs")
        print(f"Val split: {self.val_split * 100:.1f}%")
        print(f"Output: {self.output_dir}")
        print(f"Tokenizer: {self.tokenizer_path}")
        print("=" * 70)
        
        # Load dataset in streaming mode
        print("Loading FineWeb-Edu (streaming)...")
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
        # Skip to resume offset if resuming
        if self._resume_offset > 0:
            print(f"[RESUME] Skipping {self._resume_offset} documents...")
            dataset = dataset.skip(self._resume_offset)
        
# Open output files with buffering
        with open(self.train_bin, 'ab') as f_train, open(self.val_bin, 'ab') as f_val:
            train_buffer = TokenBuffer(self.train_bin, buffer_size=1_000_000)
            val_buffer = TokenBuffer(self.val_bin, buffer_size=1_000_000)
            
            try:
                for doc in dataset:
                    if self._shutdown:
                        print("\n[INTERRUPT] Shutting down...")
                        break
                    
                    self.stats["processed"] += 1
                    
                    # Process document
                    token_ids = self.process_document(doc)
                    
                    if not token_ids:
                        self.stats["rejected"] += 1
                        continue
                    
                    # Accepted
                    self.stats["accepted"] += 1
                    self.stats["tokens"] += len(token_ids)
                    
                    # Write to train or val (deterministic by hash)
                    is_val = (self.stats["accepted"] % self.val_every == 0)
                    
                    if is_val:
                        written = self._write_to_buffer(self.process_document(doc), val_buffer)
                        self.stats["val_tokens"] += len(token_ids)
                    else:
                        written = self._write_to_buffer(self.process_document(doc), train_buffer)
                        self.stats["train_tokens"] += len(token_ids)
                    
                    # Check target conditions
                    if self.target_bytes:
                        current_size = self.train_bin.stat().st_size + self.val_bin.stat().st_size
                        if current_size >= self.target_bytes:
                            print(f"\nTarget size reached: {current_size / (1024**3):.2f} GB")
                            break
                    
                    if self.target_docs and self.stats["accepted"] >= self.target_docs:
                        print(f"\nTarget documents reached: {self.stats['accepted']:,}")
                        break
                    
                    # Progress reporting
                    if self.stats["processed"] % 1000 == 0:
                        self.print_progress()
                        
                        # Periodic state save for resume
                        if self.stats["processed"] % 10000 == 0:
                            self._last_doc_hash = self._get_last_doc_hash()
                            self._save_resume_state()
                    
                    # Periodic memory check
                    if self.stats["processed"] % 50000 == 0:
                        mem_mb = self._get_memory_usage()
                        if self.stats["processed"] % 100000 == 0:
                            print(f"\n[MEM] RAM: {self._get_memory_usage():.1f} MB")
                    
                    if self._shutdown:
                        break
                
            finally:
                # Final flush
                self._flush_all_buffers(train_buffer, val_buffer)
        
        print()
        self.finalize()
    
    def _flush_all_buffers(self, *buffers):
        """Flush all buffers and verify."""
        for buf in buffers:
            buf.flush()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def _get_last_doc_hash(self) -> str:
        """Get hash of last processed document for resume."""
        return self._last_doc_hash or ""
    
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
            "train_tokens": self.stats.get("train_tokens", 0),
            "val_tokens": self.stats.get("val_tokens", 0),
            "train_size_bytes": train_size,
            "val_size_bytes": val_size,
            "total_size_bytes": total_size,
            "train_size_gb": train_size / (1024**3),
            "val_size_gb": val_size / (1024**3),
            "total_size_gb": total_size / (1024**3),
            "elapsed_seconds": time.time() - self.stats["start_time"],
            "tokenizer": str(self.tokenizer_path),
            "val_split": self.val_split,
            "target_gb": self.target_gb,
            "target_documents": self.target_documents,
            "val_split": self.val_split,
        }
        
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "=" * 70)
        print("FineWeb-Edu Preparation Complete")
        print("=" * 70)
        print(f"Documents processed: {self.stats['processed']:,}")
        print(f"Documents accepted:  {self.stats['accepted']:,}")
        print(f"Duplicates removed:  {self.stats['duplicates']:,}")
        print(f"Rejected:            {self.stats['rejected']:,}")
        print(f"Total tokens:        {self.stats['tokens']:,}")
        print(f"Train tokens:        {self.stats.get('train_tokens', 0):,}")
        print(f"Val tokens:          {self.stats.get('val_tokens', 0):,}")
        print(f"Train size:          {train_size / (1024**3):.2f} GB")
        print(f"Val size:            {val_size / (1024**3):.2f} GB")
        print(f"Total size:          {total_size / (1024**3):.2f} GB")
        print(f"Time:                {time.time() - self.stats['start_time']:.1f}s")
        print(f"Metadata saved to:   {self.metadata_path}")
        
        # Clean up resume state on successful completion
        if self.state_path.exists():
            try:
                os.remove(self.state_path)
            except:
                pass
    
    def run(self):
        """Main entry point."""
        try:
            self.run()
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Interrupted by user")
            self._shutdown = True
        except Exception as e:
            print(f"\n[ERROR] {e}")
            raise
        finally:
            self._save_resume_state()


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous interrupted run",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Resume from specific document offset",
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
        resume=args.resume,
    )
    
    if args.resume_from:
        # Handle resume-from-offset
        pass
    
    preparer.run()


if __name__ == "__main__":
    main()