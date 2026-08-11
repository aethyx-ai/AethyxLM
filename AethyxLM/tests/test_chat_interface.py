from pathlib import Path

from chat import (
    discover_checkpoints,
    newest_checkpoint,
    select_checkpoint,
    truncate_at_turn_marker,
)


def create_checkpoint_file(path: Path):
    path.write_bytes(b"checkpoint")


def test_checkpoint_discovery_is_numeric_and_prefers_latest_alias(tmp_path):
    for name in (
        "checkpoint_step_9000.pt",
        "checkpoint_step_21000.pt",
        "checkpoint_best.pt",
        "checkpoint_latest.pt",
    ):
        create_checkpoint_file(tmp_path / name)

    candidates = discover_checkpoints(tmp_path)

    assert [path.name for path in candidates] == [
        "checkpoint_latest.pt",
        "checkpoint_step_21000.pt",
        "checkpoint_step_9000.pt",
        "checkpoint_best.pt",
    ]
    assert newest_checkpoint(candidates).name == "checkpoint_latest.pt"


def test_explicit_checkpoint_selection_accepts_a_file(tmp_path):
    selected = tmp_path / "downloaded_step_21000.pt"
    create_checkpoint_file(selected)

    assert select_checkpoint(selected, tmp_path) == selected.resolve()


def test_turn_marker_truncation_does_not_rewrite_normal_text():
    normal = "This response keeps its valid spaces and punctuation."
    assert truncate_at_turn_marker(normal) == normal
    assert truncate_at_turn_marker(normal + "\nUser: next") == normal
