import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_fineweb import BinaryTokenWriter, FineWebPreparer


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer" / "tokenizer.json"


def make_documents(start=0):
    index = start
    while True:
        yield {
            "text": (
                f"Document {index} contains enough varied educational prose for "
                "streaming tokenizer verification. " * 12
            )
        }
        index += 1


def make_preparer(output_dir, target_bytes, resume=False):
    preparer = FineWebPreparer(
        target_gb=target_bytes / 1024**3,
        target_documents=None,
        val_split=0.5,
        output_dir=str(output_dir),
        tokenizer_path=str(TOKENIZER),
        resume=resume,
        resume_from=None,
        buffer_tokens=128,
        progress_seconds=3600,
        dedup_window=0,
    )
    preparer.stream_documents = lambda: make_documents(preparer.stream_offset)
    preparer._is_validation = (
        lambda _digest: preparer.stats.accepted_documents % 2 == 1
    )
    return preparer


def test_binary_writer_flushes_and_verifies_growth(tmp_path):
    path = tmp_path / "tokens.bin"
    writer = BinaryTokenWriter(path, buffer_tokens=4, append=False)
    try:
        writer.write_tokens([1, 2, 3, 4])
        assert path.stat().st_size == 0
        assert writer.should_flush
        assert writer.flush() == 4
        assert path.stat().st_size == 8
        assert np.memmap(path, dtype=np.uint16, mode="r").tolist() == [1, 2, 3, 4]
    finally:
        writer.close()


def test_interrupted_run_resumes_to_exact_target(tmp_path):
    target = 8192
    preparer = make_preparer(tmp_path, target)

    def interrupted_stream():
        documents = make_documents()
        for index in range(4):
            if index == 3:
                preparer._stop_requested = True
            yield next(documents)

    preparer.stream_documents = interrupted_stream
    assert preparer.run() == "interrupted"

    train_path = tmp_path / "fineweb_train.bin"
    val_path = tmp_path / "fineweb_val.bin"
    first_size = train_path.stat().st_size + val_path.stat().st_size
    assert 0 < first_size < target
    assert train_path.stat().st_size > 0
    assert val_path.stat().st_size > 0
    first_train = train_path.read_bytes()

    resumed = make_preparer(tmp_path, target, resume=True)
    assert resumed.run() == "complete"
    assert train_path.stat().st_size + val_path.stat().st_size == target
    assert train_path.read_bytes().startswith(first_train)

    state = json.loads((tmp_path / "fineweb_state.json").read_text())
    metadata = json.loads((tmp_path / "fineweb_metadata.json").read_text())
    assert state["train_size_bytes"] == train_path.stat().st_size
    assert state["val_size_bytes"] == val_path.stat().st_size
    assert metadata["total_size_bytes"] == target
    assert metadata["status"] == "complete"

    with pytest.raises(ValueError, match="already reached its target"):
        make_preparer(tmp_path, target * 2, resume=True)


def test_interruption_flushes_usable_files(tmp_path):
    preparer = make_preparer(tmp_path, 1_000_000)

    def interrupted_stream():
        documents = make_documents()
        for index in range(10):
            if index == 5:
                preparer._stop_requested = True
            yield next(documents)

    preparer.stream_documents = interrupted_stream
    assert preparer.run() == "interrupted"

    train_size = (tmp_path / "fineweb_train.bin").stat().st_size
    val_size = (tmp_path / "fineweb_val.bin").stat().st_size
    assert train_size + val_size > 0
    assert train_size % np.dtype(np.uint16).itemsize == 0
    assert val_size % np.dtype(np.uint16).itemsize == 0

    state = json.loads((tmp_path / "fineweb_state.json").read_text())
    assert state["train_size_bytes"] == train_size
    assert state["val_size_bytes"] == val_size
