import json
import sys
import types
from pathlib import Path

import pytest

from training.checkpoint_backup import (
    KaggleDatasetCheckpointBackup,
    create_checkpoint_backup,
)


def test_kaggle_backup_stages_only_current_checkpoint(tmp_path, monkeypatch):
    uploads = []
    fake_kagglehub = types.SimpleNamespace(
        dataset_upload=lambda handle, directory, version_notes: uploads.append(
            (handle, Path(directory), version_notes)
        )
    )
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)
    checkpoint = tmp_path / "checkpoint_step_41000.pt"
    checkpoint.write_bytes(b"checkpoint")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "checkpoint_step_40000.pt").write_bytes(b"old")

    backup = KaggleDatasetCheckpointBackup(
        "aethyx/aethyxlm-live-checkpoints", staging, retries=1
    )

    assert backup.upload(checkpoint, 41000)
    assert not (staging / "checkpoint_step_40000.pt").exists()
    assert (staging / checkpoint.name).read_bytes() == b"checkpoint"
    manifest = json.loads((staging / "backup_manifest.json").read_text())
    assert manifest["step"] == 41000
    assert uploads[0][0] == "aethyx/aethyxlm-live-checkpoints"
    assert "41000" in uploads[0][2]


def test_required_backup_raises_after_retry(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setitem(
        sys.modules, "kagglehub", types.SimpleNamespace(dataset_upload=fail)
    )
    checkpoint = tmp_path / "checkpoint_step_42000.pt"
    checkpoint.write_bytes(b"checkpoint")
    backup = KaggleDatasetCheckpointBackup(
        "aethyx/aethyxlm-live-checkpoints",
        tmp_path / "staging",
        required=True,
        retries=1,
    )

    with pytest.raises(RuntimeError, match="Persistent checkpoint backup failed"):
        backup.upload(checkpoint, 42000)


def test_disabled_backup_returns_none(tmp_path):
    assert create_checkpoint_backup({"enabled": False}, tmp_path) is None
