import hashlib
from pathlib import Path

from chat import (
    discover_checkpoints,
    format_inference_prompt,
    newest_checkpoint,
    resolve_tokenizer_path,
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


def test_inference_prompt_modes_are_explicit():
    assert format_inference_prompt("The capital of India is", "base") == "The capital of India is"
    assert format_inference_prompt("What is the capital?", "chat") == (
        "User: What is the capital?\nAethyx:"
    )
    assert format_inference_prompt("Next", "chat", "User: First\nAethyx: Answer\n") == (
        "User: First\nAethyx: Answer\nUser: Next\nAethyx:"
    )


def test_tokenizer_is_detected_from_checkpoint_metadata(tmp_path):
    checkpoint = tmp_path / "checkpoint_step_30000.pt"
    tokenizer = tmp_path / "tokenizer_v3_48k.json"
    create_checkpoint_file(checkpoint)
    tokenizer.write_bytes(b"v3 tokenizer")
    digest = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    config = {
        "tokenizer": {
            "file_name": tokenizer.name,
            "sha256": digest,
        }
    }

    assert resolve_tokenizer_path(checkpoint, config) == tokenizer.resolve()


def test_tokenizer_detection_uses_fingerprint_when_file_was_renamed(tmp_path):
    checkpoint = tmp_path / "checkpoint_step_30000.pt"
    tokenizer = tmp_path / "renamed.json"
    create_checkpoint_file(checkpoint)
    tokenizer.write_bytes(b"v3 tokenizer")
    digest = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    config = {
        "tokenizer": {
            "file_name": "tokenizer_v3_48k.json",
            "sha256": digest,
        }
    }

    assert resolve_tokenizer_path(checkpoint, config) == tokenizer.resolve()
