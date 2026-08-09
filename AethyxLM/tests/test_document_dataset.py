from pathlib import Path

import numpy as np

from dataset.dataset import AethyxDataset, MixedAethyxDataset
from tokenizer.tokenizer import AethyxTokenizer


def test_text_tokenization_inserts_document_boundaries(tmp_path: Path):
    source = tmp_path / "documents.txt"
    source.write_text("first document\n\nsecond document\n", encoding="utf-8")
    dataset = AethyxDataset(source, context_length=2)
    tokenizer = AethyxTokenizer()
    assert tokenizer.eos_id is not None
    assert np.count_nonzero(dataset._data == tokenizer.eos_id) == 2


def test_mixed_training_dataset_does_not_include_validation(tmp_path: Path):
    train = tmp_path / "train.bin"
    validation = tmp_path / "validation.bin"
    np.arange(20, dtype=np.uint16).tofile(train)
    np.arange(100, 120, dtype=np.uint16).tofile(validation)

    dataset = MixedAethyxDataset(
        [{"train": str(train), "val": str(validation), "weight": 1.0}],
        context_length=4,
    )
    assert len(dataset.sub_datasets) == 1
    assert Path(dataset.sub_datasets[0]._data.filename) == train


def test_mixed_sampling_is_reproducible_and_epoch_aware(tmp_path: Path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    np.arange(100, dtype=np.uint16).tofile(first)
    np.arange(1000, 1100, dtype=np.uint16).tofile(second)
    config = [
        {"train": str(first), "weight": 0.5},
        {"train": str(second), "weight": 0.5},
    ]
    left = MixedAethyxDataset(config, context_length=4, seed=9)
    right = MixedAethyxDataset(config, context_length=4, seed=9)
    assert len(left) == sum(len(item) for item in left.sub_datasets)
    assert [left[index][0].tolist() for index in range(10)] == [
        right[index][0].tolist() for index in range(10)
    ]
    before = [left[index][0].tolist() for index in range(10)]
    left.set_epoch(1)
    after = [left[index][0].tolist() for index in range(10)]
    assert before != after
