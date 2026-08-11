import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_checkpoint_suite.py"
SPEC = importlib.util.spec_from_file_location("evaluate_checkpoint_suite", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _checkpoint(path, step):
    torch.save({"step": step, "model_state_dict": {"x": torch.ones(1)}}, path)


def test_unique_checkpoint_steps_prefers_numbered_files_over_aliases(tmp_path):
    latest = tmp_path / "checkpoint_latest.pt"
    numbered = tmp_path / "checkpoint_step_37000.pt"
    best = tmp_path / "checkpoint_best.pt"
    older = tmp_path / "checkpoint_step_36000.pt"
    _checkpoint(latest, 37000)
    _checkpoint(numbered, 37000)
    _checkpoint(best, 36000)
    _checkpoint(older, 36000)

    result = MODULE.unique_checkpoint_steps([latest, best, numbered, older])

    assert result == [older, numbered]
