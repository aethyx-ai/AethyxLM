from pathlib import Path

from train import PROJECT_ROOT, resolve_project_path


def test_relative_checkpoint_directory_is_repository_rooted():
    assert resolve_project_path("checkpoints") == PROJECT_ROOT / "checkpoints"


def test_absolute_checkpoint_directory_is_preserved(tmp_path: Path):
    assert resolve_project_path(str(tmp_path)) == tmp_path.resolve()
