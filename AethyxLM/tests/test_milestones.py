import json

import pytest
import torch

from training.milestones import archive_milestone


def test_archive_milestone_is_non_rotating_and_manifested(tmp_path):
    source = tmp_path / "checkpoint_latest.pt"
    torch.save(
        {
            "step": 40000,
            "tokens_seen": 123,
            "best_val_loss": 2.0,
            "model_state_dict": {"weight": torch.ones(2)},
            "config": {"model": {"num_layers": 12}, "tokenizer_sha256": "abc"},
        },
        source,
    )

    destination = archive_milestone(source, tmp_path / "milestones")

    assert destination.name == "checkpoint_step_40000.pt"
    manifest = json.loads((destination.parent / "manifest.json").read_text())
    assert manifest["milestones"][0]["step"] == 40000
    assert manifest["milestones"][0]["tokenizer_sha256"] == "abc"
    assert len(manifest["milestones"][0]["sha256"]) == 64


def test_existing_milestone_requires_identical_content(tmp_path):
    source = tmp_path / "checkpoint_latest.pt"
    checkpoint = {
        "step": 40000,
        "model_state_dict": {"weight": torch.ones(2)},
        "config": {},
    }
    torch.save(checkpoint, source)
    archive_milestone(source, tmp_path / "milestones")

    checkpoint["model_state_dict"]["weight"] = torch.zeros(2)
    torch.save(checkpoint, source)
    with pytest.raises(FileExistsError, match="different content"):
        archive_milestone(source, tmp_path / "milestones")
