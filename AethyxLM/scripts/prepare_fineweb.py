#!/usr/bin/env python3
"""Stream FineWeb-Edu into verified uint16 training and validation files."""

import argparse
import ctypes
import gc
import hashlib
import json
import os
import re
import signal
import sys
import time
import unicodedata
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.tokenizer import AethyxTokenizer


BYTES_PER_TOKEN = np.dtype("<u2").itemsize
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
STATE_VERSION = 2
STREAM_BATCH_SIZE = 100
SPACE_RE = re.compile(r"[ \t]+")
NEWLINE_RE = re.compile(r"\n{3,}")


@dataclass
class PrepStats:
    processed_documents: int = 0
    accepted_documents: int = 0
    rejected_documents: int = 0
    duplicate_documents: int = 0
    train_tokens: int = 0
    val_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.train_tokens + self.val_tokens


class BinaryTokenWriter:
    """A bounded, append-only uint16 writer with exact size verification."""

    def __init__(self, path: Path, buffer_tokens: int, append: bool) -> None:
        if buffer_tokens <= 0:
            raise ValueError("buffer_tokens must be positive")

        self.path = path
        self.buffer_tokens = buffer_tokens
        self._buffer: List[int] = []
        self._file = path.open("ab" if append else "wb", buffering=1024 * 1024)
        self.expected_size = path.stat().st_size
        if self.expected_size % BYTES_PER_TOKEN:
            self._file.close()
            raise ValueError(f"{path} has a partial uint16 token ({self.expected_size} bytes)")

    @property
    def buffered_tokens(self) -> int:
        return len(self._buffer)

    @property
    def total_tokens(self) -> int:
        return self.expected_size // BYTES_PER_TOKEN + len(self._buffer)

    @property
    def should_flush(self) -> bool:
        return len(self._buffer) >= self.buffer_tokens

    def write_tokens(self, token_ids: List[int], limit: Optional[int] = None) -> int:
        count = len(token_ids) if limit is None else min(len(token_ids), limit)
        if count <= 0:
            return 0
        if count == len(token_ids):
            self._buffer.extend(token_ids)
        else:
            self._buffer.extend(token_ids[:count])
        return count

    def flush(self) -> int:
        if not self._buffer:
            self._verify_size()
            return 0

        tokens = np.asarray(self._buffer, dtype="<u2")
        previous_size = self.expected_size
        new_size = previous_size + int(tokens.nbytes)
        try:
            tokens.tofile(self._file)
            self._file.flush()
            os.fsync(self._file.fileno())
            self.expected_size = new_size
            self._verify_size()
        except Exception:
            actual_size = os.path.getsize(self.path)
            if actual_size == new_size:
                self.expected_size = new_size
                self._buffer.clear()
            elif actual_size != previous_size:
                self._file.seek(previous_size)
                self._file.truncate()
                self._file.flush()
                os.fsync(self._file.fileno())
            raise

        count = len(self._buffer)
        self._buffer.clear()
        return count

    def _verify_size(self) -> None:
        actual_size = os.path.getsize(self.path)
        if actual_size != self.expected_size:
            raise IOError(
                f"Write verification failed for {self.path}: "
                f"expected {self.expected_size:,} bytes, found {actual_size:,}"
            )

    def close(self) -> None:
        self._file.close()


class FineWebPreparer:
    """Coordinate streaming, tokenization, splitting, and durable writes."""

    def __init__(
        self,
        target_gb: Optional[float],
        target_documents: Optional[int],
        val_split: float,
        output_dir: str,
        tokenizer_path: str,
        resume: bool,
        resume_from: Optional[int],
        buffer_tokens: int,
        progress_seconds: float,
        dedup_window: int,
    ) -> None:
        if target_gb is None and target_documents is None:
            raise ValueError("Either target_gb or target_documents is required")
        if target_gb is not None and target_gb <= 0:
            raise ValueError("target_gb must be positive")
        if target_documents is not None and target_documents <= 0:
            raise ValueError("target_documents must be positive")
        if not 0.0 <= val_split <= 1.0:
            raise ValueError("val_split must be between 0 and 1")
        if resume_from is not None and resume_from < 0:
            raise ValueError("resume_from must be non-negative")
        if dedup_window < 0:
            raise ValueError("dedup_window must be non-negative")

        self.target_gb = target_gb
        self.target_documents = target_documents
        self.requested_target_bytes = (
            int(target_gb * 1024**3) if target_gb is not None else None
        )
        self.target_tokens = (
            self.requested_target_bytes // BYTES_PER_TOKEN
            if self.requested_target_bytes is not None
            else None
        )
        if self.target_tokens == 0:
            raise ValueError("target_gb is too small to contain one uint16 token")
        self.target_bytes = (
            self.target_tokens * BYTES_PER_TOKEN
            if self.target_tokens is not None
            else None
        )
        self.val_split = val_split
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path = Path(tokenizer_path)
        self.tokenizer = AethyxTokenizer(self.tokenizer_path)
        self.tokenizer_sha256 = sha256_file(self.tokenizer_path)
        if self.tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
            raise ValueError(
                f"Vocabulary size {self.tokenizer.vocab_size:,} does not fit uint16"
            )

        self.train_path = self.output_dir / "fineweb_train.bin"
        self.val_path = self.output_dir / "fineweb_val.bin"
        self.state_path = self.output_dir / "fineweb_state.json"
        self.metadata_path = self.output_dir / "fineweb_metadata.json"
        self.resume = resume
        self.resume_from = resume_from
        self.buffer_tokens = buffer_tokens
        self.progress_seconds = progress_seconds
        self.dedup_window = dedup_window

        self.stats = PrepStats()
        self.stream_offset = 0
        self.created_at = datetime.now(timezone.utc).isoformat()
        self._seen_hashes: Set[bytes] = set()
        self._hash_order: Deque[bytes] = deque()
        self._stop_requested = False
        self._start_time = 0.0
        self._last_report_time = 0.0
        self._session_start_tokens = 0
        self._session_start_bytes = 0
        self._run_status = "running"
        self._last_memory_release = 0.0

        if resume:
            self._load_resume_state()
        elif resume_from is not None:
            self.stream_offset = resume_from

    def stream_documents(self) -> Iterator[Dict[str, object]]:
        """Yield documents from Hugging Face without materializing the dataset."""
        dataset = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split="train",
            streaming=True,
            columns=["text"],
            batch_size=STREAM_BATCH_SIZE,
        )
        if self.stream_offset:
            print(f"Skipping {self.stream_offset:,} streamed documents for resume...")
            dataset = dataset.skip(self.stream_offset)
        yield from dataset

    @staticmethod
    def clean_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = SPACE_RE.sub(" ", text)
        return NEWLINE_RE.sub("\n\n", text).strip()

    @staticmethod
    def _valid_text(text: str) -> bool:
        return len(text) >= 100 and len(set(text)) >= 10

    def _is_duplicate(self, digest: bytes) -> bool:
        if not self.dedup_window:
            return False
        if digest in self._seen_hashes:
            return True
        if len(self._hash_order) >= self.dedup_window:
            expired = self._hash_order.popleft()
            self._seen_hashes.remove(expired)
        self._hash_order.append(digest)
        self._seen_hashes.add(digest)
        return False

    def tokenize_document(
        self, document: Dict[str, object]
    ) -> Tuple[Optional[List[int]], Optional[bytes], bool]:
        """Clean and tokenize one document; no token data survives the iteration."""
        raw_text = document.get("text")
        if not isinstance(raw_text, str):
            return None, None, False

        text = self.clean_text(raw_text)
        if not self._valid_text(text):
            return None, None, False

        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        if self._is_duplicate(digest):
            return None, digest, True

        try:
            token_ids = self.tokenizer.encode(text)
        except Exception as exc:
            print(f"\nTokenizer rejected a document: {exc}", file=sys.stderr)
            return None, digest, False
        if not token_ids:
            return None, digest, False
        if min(token_ids) < 0 or max(token_ids) > np.iinfo(np.uint16).max:
            raise ValueError("Tokenizer emitted an ID outside the uint16 range")
        return token_ids, digest, False

    def _is_validation(self, digest: bytes) -> bool:
        if self.val_split <= 0:
            return False
        if self.val_split >= 1:
            return True
        value = int.from_bytes(digest, "big") / float(1 << 64)
        return value < self.val_split

    def write_tokens(
        self,
        token_ids: List[int],
        digest: bytes,
        train_writer: BinaryTokenWriter,
        val_writer: BinaryTokenWriter,
    ) -> int:
        """Route one document directly into its bounded split buffer."""
        remaining = None
        if self.target_tokens is not None:
            remaining = self.target_tokens - (
                train_writer.total_tokens + val_writer.total_tokens
            )
            if remaining <= 0:
                return 0

        if self._is_validation(digest):
            written = val_writer.write_tokens(token_ids, remaining)
            self.stats.val_tokens += written
        else:
            written = train_writer.write_tokens(token_ids, remaining)
            self.stats.train_tokens += written
        return written

    def flush_buffers(
        self,
        train_writer: BinaryTokenWriter,
        val_writer: BinaryTokenWriter,
        save_state: bool = True,
    ) -> None:
        """Synchronously flush both splits and checkpoint their matching offset."""
        train_writer.flush()
        val_writer.flush()
        if save_state:
            self._save_resume_state(train_writer, val_writer)
        now = time.monotonic()
        if now - self._last_memory_release >= 30.0:
            release_unused_memory()
            self._last_memory_release = now

    def _target_reached(
        self, train_writer: BinaryTokenWriter, val_writer: BinaryTokenWriter
    ) -> bool:
        if self.target_tokens is not None:
            if train_writer.total_tokens + val_writer.total_tokens >= self.target_tokens:
                return True
        if self.target_documents is not None:
            return self.stats.accepted_documents >= self.target_documents
        return False

    def progress_report(
        self,
        train_writer: BinaryTokenWriter,
        val_writer: BinaryTokenWriter,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now - self._last_report_time < self.progress_seconds:
            return
        self._last_report_time = now

        elapsed = max(now - self._start_time, 1e-9)
        train_size = os.path.getsize(self.train_path)
        val_size = os.path.getsize(self.val_path)
        committed_size = train_size + val_size
        session_tokens = self.stats.total_tokens - self._session_start_tokens
        session_bytes = committed_size - self._session_start_bytes
        token_speed = session_tokens / elapsed
        write_speed = session_bytes / elapsed

        eta = None
        if self.target_bytes is not None and write_speed > 0:
            eta = max(0.0, self.target_bytes - committed_size) / write_speed

        print(
            "\n"
            f"Processed Docs : {self.stats.processed_documents:,}\n"
            f"Accepted Docs  : {self.stats.accepted_documents:,}\n"
            f"Rejected Docs  : {self.stats.rejected_documents:,}\n"
            f"Duplicates     : {self.stats.duplicate_documents:,}\n"
            f"Train Tokens   : {format_count(self.stats.train_tokens)}\n"
            f"Val Tokens     : {format_count(self.stats.val_tokens)}\n"
            f"Train Size     : {format_bytes(train_size)}"
            f" (+{format_count(train_writer.buffered_tokens)} buffered tokens)\n"
            f"Val Size       : {format_bytes(val_size)}"
            f" (+{format_count(val_writer.buffered_tokens)} buffered tokens)\n"
            f"Write Speed    : {format_bytes(write_speed)}/s\n"
            f"Token Speed    : {format_count(token_speed)} tok/s\n"
            f"Memory RSS     : {format_memory(memory_usage_mb())}\n"
            f"Elapsed        : {format_duration(elapsed)}\n"
            f"ETA            : {format_duration(eta) if eta is not None else 'N/A'}",
            flush=True,
        )

    def save_metadata(self, status: str) -> None:
        duration = max(0.0, time.monotonic() - self._start_time)
        train_size = self.train_path.stat().st_size
        val_size = self.val_path.stat().st_size
        metadata = {
            "dataset": DATASET_NAME,
            "dataset_config": DATASET_CONFIG,
            "tokenizer": str(self.tokenizer_path),
            "vocabulary_size": self.tokenizer.vocab_size,
            "token_dtype": "uint16",
            "total_documents": self.stats.processed_documents,
            "accepted_documents": self.stats.accepted_documents,
            "rejected_documents": self.stats.rejected_documents,
            "duplicate_documents": self.stats.duplicate_documents,
            "train_tokens": self.stats.train_tokens,
            "validation_tokens": self.stats.val_tokens,
            "train_size_bytes": train_size,
            "validation_size_bytes": val_size,
            "total_size_bytes": train_size + val_size,
            "target_size_bytes": self.target_bytes,
            "requested_target_size_bytes": self.requested_target_bytes,
            "target_size_gb": self.target_gb,
            "validation_split": self.val_split,
            "stream_batch_size": STREAM_BATCH_SIZE,
            "creation_date": self.created_at,
            "completed_date": datetime.now(timezone.utc).isoformat(),
            "preparation_duration_seconds": duration,
            "status": status,
        }
        atomic_write_json(self.metadata_path, metadata)

    def _load_resume_state(self) -> None:
        if not self.state_path.exists():
            raise FileNotFoundError(
                f"Cannot resume without {self.state_path}. Use --resume-from with a "
                "known source offset, or omit --resume to start new files."
            )
        with self.state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"Unsupported resume state version in {self.state_path}")
        if state.get("status") == "complete":
            raise ValueError(
                "This output already reached its target. Exact-size completion may "
                "truncate the final document, so only interrupted/error runs can resume."
            )
        expected_config = {
            "dataset": DATASET_NAME,
            "dataset_config": DATASET_CONFIG,
            "tokenizer_sha256": self.tokenizer_sha256,
            "vocabulary_size": self.tokenizer.vocab_size,
            "validation_split": self.val_split,
            "stream_batch_size": STREAM_BATCH_SIZE,
            "dedup_window": self.dedup_window,
        }
        for key, expected in expected_config.items():
            if state.get(key) != expected:
                raise ValueError(
                    f"Resume setting mismatch for {key}: "
                    f"checkpoint={state.get(key)!r}, requested={expected!r}"
                )

        self.stats = PrepStats(**state["stats"])
        self.stream_offset = int(state["stream_offset"])
        self.created_at = state.get("created_at", self.created_at)
        expected_train = int(state["train_size_bytes"])
        expected_val = int(state["val_size_bytes"])
        actual_train = self.train_path.stat().st_size if self.train_path.exists() else 0
        actual_val = self.val_path.stat().st_size if self.val_path.exists() else 0
        if (actual_train, actual_val) != (expected_train, expected_val):
            raise IOError(
                "Resume files do not match the last durable checkpoint: "
                f"train expected/actual={expected_train}/{actual_train}, "
                f"val expected/actual={expected_val}/{actual_val}"
            )
        if self.stats.train_tokens != actual_train // BYTES_PER_TOKEN:
            raise ValueError("Resume train token count does not match train file size")
        if self.stats.val_tokens != actual_val // BYTES_PER_TOKEN:
            raise ValueError("Resume validation token count does not match val file size")
        if self.target_tokens is not None:
            existing_tokens = self.stats.train_tokens + self.stats.val_tokens
            if existing_tokens > self.target_tokens:
                raise ValueError("Resume target is smaller than the existing output")
        if (
            self.target_documents is not None
            and self.stats.accepted_documents > self.target_documents
        ):
            raise ValueError("Resume document target is smaller than the checkpoint")

        hashes = [bytes.fromhex(value) for value in state.get("dedup_hashes", [])]
        if len(hashes) > self.dedup_window:
            raise ValueError("Resume deduplication state exceeds the configured window")
        self._hash_order.extend(hashes)
        self._seen_hashes.update(hashes)

    def _save_resume_state(
        self, train_writer: BinaryTokenWriter, val_writer: BinaryTokenWriter
    ) -> None:
        if train_writer.buffered_tokens or val_writer.buffered_tokens:
            raise RuntimeError("Refusing to checkpoint while tokens are still buffered")
        state = {
            "version": STATE_VERSION,
            "dataset": DATASET_NAME,
            "dataset_config": DATASET_CONFIG,
            "tokenizer_sha256": self.tokenizer_sha256,
            "vocabulary_size": self.tokenizer.vocab_size,
            "validation_split": self.val_split,
            "stream_batch_size": STREAM_BATCH_SIZE,
            "dedup_window": self.dedup_window,
            "dedup_hashes": [value.hex() for value in self._hash_order],
            "stream_offset": self.stream_offset,
            "stats": asdict(self.stats),
            "train_size_bytes": train_writer.expected_size,
            "val_size_bytes": val_writer.expected_size,
            "created_at": self.created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": self._run_status,
        }
        atomic_write_json(self.state_path, state)

    def _signal_handler(self, signum: int, _frame: object) -> None:
        if self._stop_requested:
            raise KeyboardInterrupt
        self._stop_requested = True
        print(f"\nSignal {signum} received; stopping after the current document...")

    def run(self) -> str:
        append = self.resume
        train_writer = BinaryTokenWriter(self.train_path, self.buffer_tokens, append)
        val_writer = BinaryTokenWriter(self.val_path, self.buffer_tokens, append)
        status = "running"
        old_sigint = signal.getsignal(signal.SIGINT)
        old_sigterm = signal.getsignal(signal.SIGTERM)
        old_sigbreak = (
            signal.getsignal(signal.SIGBREAK) if hasattr(signal, "SIGBREAK") else None
        )
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self._signal_handler)

        self._start_time = time.monotonic()
        self._last_report_time = self._start_time - self.progress_seconds
        self._session_start_tokens = self.stats.total_tokens
        self._session_start_bytes = (
            train_writer.expected_size + val_writer.expected_size
        )
        self._print_header()

        try:
            self.flush_buffers(train_writer, val_writer)
            if self._target_reached(train_writer, val_writer):
                status = "complete"
            else:
                for document in self.stream_documents():
                    if self._stop_requested:
                        status = "interrupted"
                        break

                    token_ids, digest, duplicate = self.tokenize_document(document)
                    self.stream_offset += 1
                    self.stats.processed_documents += 1
                    if token_ids is None or digest is None:
                        self.stats.rejected_documents += 1
                        if duplicate:
                            self.stats.duplicate_documents += 1
                    else:
                        written = self.write_tokens(
                            token_ids, digest, train_writer, val_writer
                        )
                        if written:
                            self.stats.accepted_documents += 1
                        else:
                            status = "complete"
                            break

                    if train_writer.should_flush or val_writer.should_flush:
                        self.flush_buffers(train_writer, val_writer)
                    self.progress_report(train_writer, val_writer)

                    if self._target_reached(train_writer, val_writer):
                        status = "complete"
                        break
                else:
                    status = "dataset_exhausted"
        except KeyboardInterrupt:
            status = "interrupted"
            self._stop_requested = True
        except Exception:
            status = "error"
            raise
        finally:
            self._run_status = status
            try:
                self.flush_buffers(train_writer, val_writer)
                self.progress_report(train_writer, val_writer, force=True)
                self.save_metadata(status)
            finally:
                train_writer.close()
                val_writer.close()
                signal.signal(signal.SIGINT, old_sigint)
                signal.signal(signal.SIGTERM, old_sigterm)
                if hasattr(signal, "SIGBREAK"):
                    signal.signal(signal.SIGBREAK, old_sigbreak)

        print(f"\nPreparation status: {status}")
        print(f"Metadata: {self.metadata_path}")
        return status

    def _print_header(self) -> None:
        target = (
            f"{self.target_gb:g} GiB ({self.target_tokens:,} uint16 tokens)"
            if self.target_gb is not None
            else f"{self.target_documents:,} accepted documents"
        )
        print("=" * 72)
        print("FineWeb-Edu streaming preparation")
        print(f"Target          : {target}")
        print(f"Validation split: {self.val_split:.2%}")
        print(f"Buffer per split: {self.buffer_tokens:,} tokens")
        print(f"Train output    : {self.train_path}")
        print(f"Validation output: {self.val_path}")
        print(f"Resume offset   : {self.stream_offset:,}")
        print("=" * 72)


def atomic_write_json(path: Path, value: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def memory_usage_mb() -> Optional[float]:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        pass

    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        if ctypes.windll.psapi.GetProcessMemoryInfo(
            get_process(), ctypes.byref(counters), counters.cb
        ):
            return counters.WorkingSetSize / 1024**2
    return None


def release_unused_memory() -> None:
    """Return unreachable Python and Arrow allocations after durable flushes."""
    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except Exception:
        pass


def format_count(value: float) -> str:
    for suffix, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.2f}{suffix}"
    return f"{value:,.0f}"


def format_bytes(value: float) -> str:
    for suffix, scale in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if abs(value) >= scale:
            return f"{value / scale:.2f} {suffix}"
    return f"{value:,.0f} B"


def format_memory(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.1f} MiB"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream FineWeb-Edu into uint16 binary training files"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-gb", type=float, help="Combined output size in GiB")
    target.add_argument(
        "--target-documents", type=int, help="Accepted document target"
    )
    parser.add_argument("--val-split", type=float, default=0.01)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument(
        "--resume", action="store_true", help="Append from the durable state checkpoint"
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        help="Start new output files after skipping this many source documents",
    )
    parser.add_argument("--buffer-tokens", type=int, default=500_000)
    parser.add_argument("--progress-seconds", type=float, default=5.0)
    parser.add_argument(
        "--dedup-window",
        type=int,
        default=0,
        help="Bounded recent-document hash window; use 0 to disable",
    )
    args = parser.parse_args()
    if args.resume and args.resume_from is not None:
        parser.error("--resume and --resume-from cannot be used together")
    return args


def main() -> None:
    args = parse_args()
    preparer = FineWebPreparer(
        target_gb=args.target_gb,
        target_documents=args.target_documents,
        val_split=args.val_split,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        resume=args.resume,
        resume_from=args.resume_from,
        buffer_tokens=args.buffer_tokens,
        progress_seconds=args.progress_seconds,
        dedup_window=args.dedup_window,
    )
    preparer.run()


if __name__ == "__main__":
    main()
