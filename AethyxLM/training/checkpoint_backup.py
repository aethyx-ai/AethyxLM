"""Persistent checkpoint backups for cloud training environments."""

import json
import os
import shutil
import time
from pathlib import Path


class KaggleDatasetCheckpointBackup:
    """Upload one checkpoint per private Kaggle Dataset version."""

    def __init__(
        self,
        handle: str,
        staging_dir: Path,
        required: bool = True,
        retries: int = 3,
    ):
        if "/" not in handle:
            raise ValueError("Kaggle backup handle must be '<owner>/<dataset>'")
        self.handle = handle
        self.staging_dir = Path(staging_dir).expanduser().resolve()
        self.required = bool(required)
        self.retries = max(1, int(retries))

    def _prepare_staging(self, checkpoint_path: Path, step: int) -> Path:
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.staging_dir.iterdir():
            if stale.is_file() and (
                (stale.name.startswith("checkpoint_") and stale.suffix == ".pt")
                or stale.name == "backup_manifest.json"
            ):
                stale.unlink()

        staged_checkpoint = self.staging_dir / checkpoint_path.name
        try:
            os.link(checkpoint_path, staged_checkpoint)
        except OSError:
            shutil.copy2(checkpoint_path, staged_checkpoint)

        manifest = {
            "step": int(step),
            "checkpoint": checkpoint_path.name,
            "dataset": self.handle,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (self.staging_dir / "backup_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return staged_checkpoint

    def upload(self, checkpoint_path: Path, step: int) -> bool:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        staged_checkpoint = self._prepare_staging(checkpoint_path, step)

        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                import kagglehub

                print(
                    f"[Backup] Uploading {staged_checkpoint.name} to "
                    f"{self.handle} (attempt {attempt}/{self.retries})...",
                    flush=True,
                )
                kagglehub.dataset_upload(
                    self.handle,
                    str(self.staging_dir),
                    version_notes=f"AethyxLM checkpoint at optimizer step {step}",
                )
                print(
                    f"[Backup] Persistent Kaggle backup complete for step {step}",
                    flush=True,
                )
                return True
            except Exception as error:  # the provider raises several SDK error types
                last_error = error
                print(f"[Backup] Attempt {attempt} failed: {error}", flush=True)
                if attempt < self.retries:
                    time.sleep(min(30, 5 * attempt))

        message = (
            f"Persistent checkpoint backup failed after {self.retries} attempts: "
            f"{last_error}"
        )
        if self.required:
            raise RuntimeError(message) from last_error
        print(f"[Backup][WARN] {message}", flush=True)
        return False


def create_checkpoint_backup(config, checkpoint_dir: Path):
    """Create the configured provider, or return ``None`` when disabled."""
    if not config or not config.get("enabled", False):
        return None
    provider = config.get("provider", "kaggle_dataset")
    if provider != "kaggle_dataset":
        raise ValueError(f"Unsupported checkpoint backup provider: {provider}")
    handle = config.get("handle")
    if not handle:
        raise ValueError("checkpoint backup requires a Kaggle Dataset handle")
    staging_dir = config.get("staging_dir") or Path(checkpoint_dir) / ".kaggle_backup"
    return KaggleDatasetCheckpointBackup(
        handle=handle,
        staging_dir=Path(staging_dir),
        required=config.get("required", True),
        retries=config.get("retries", 3),
    )
