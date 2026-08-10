import json

import pytest

from scripts.prepare_dataset_bundle import completed_output
from scripts.prepare_fineweb import BinaryTokenWriter, FineWebPreparer


def test_generic_preparer_uses_isolated_prefix_and_refuses_overwrite(tmp_path):
    tokenizer = "tokenizer/tokenizer.json"
    preparer = FineWebPreparer(
        target_gb=0.001,
        target_documents=None,
        val_split=0.01,
        output_dir=str(tmp_path),
        tokenizer_path=tokenizer,
        resume=False,
        resume_from=None,
        buffer_tokens=10,
        progress_seconds=1,
        dedup_window=0,
        dataset_name="example/source",
        dataset_config=None,
        source_split="train",
        text_field="content",
        output_prefix="example_v1",
    )
    assert preparer.train_path.name == "example_v1_train.bin"
    preparer.train_path.touch()
    with pytest.raises(FileExistsError):
        FineWebPreparer(
            target_gb=0.001,
            target_documents=None,
            val_split=0.01,
            output_dir=str(tmp_path),
            tokenizer_path=tokenizer,
            resume=False,
            resume_from=None,
            buffer_tokens=10,
            progress_seconds=1,
            dedup_window=0,
            output_prefix="example_v1",
        )


def test_binary_writer_and_completion_check_verify_physical_sizes(tmp_path):
    train = tmp_path / "source_train.bin"
    val = tmp_path / "source_val.bin"
    writer = BinaryTokenWriter(train, buffer_tokens=2, append=False)
    writer.write_tokens([1, 2, 3])
    writer.flush()
    writer.close()
    val.write_bytes(b"\x04\x00")
    metadata = {
        "status": "complete",
        "train_size_bytes": 6,
        "validation_size_bytes": 2,
        "total_size_bytes": 8,
    }
    (tmp_path / "source_metadata.json").write_text(json.dumps(metadata))
    assert completed_output({"name": "source"}, tmp_path)
    val.write_bytes(b"")
    assert not completed_output({"name": "source"}, tmp_path)
