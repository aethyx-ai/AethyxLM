"""Immutable milestone checkpoint archival and manifest management."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint_summary(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Not an AethyxLM training checkpoint: {path}")
    model = checkpoint.get("config", {}).get("model", {})
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return {
        "step": int(checkpoint.get("step", -1)),
        "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
        "tokenizer_sha256": (
            checkpoint.get("config", {}).get("tokenizer", {}).get("sha256")
            or checkpoint.get("config", {}).get("tokenizer_sha256")
        ),
        "model": model,
    }


def archive_milestone(
    source: str | Path,
    milestone_dir: str | Path,
    label: str | None = None,
) -> Path:
    """Copy a checkpoint atomically into a non-rotating milestone directory."""
    source = Path(source).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint is missing or empty: {source}")
    summary = read_checkpoint_summary(source)
    if summary["step"] < 0:
        raise RuntimeError("Checkpoint does not declare a valid training step")

    destination_dir = Path(milestone_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""
    destination = destination_dir / f"checkpoint_step_{summary['step']}{suffix}.pt"
    if destination.exists():
        if (
            destination.stat().st_size != source.stat().st_size
            or sha256_file(destination) != sha256_file(source)
        ):
            raise FileExistsError(
                f"Milestone already exists with different content: {destination}"
            )
    else:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    record = {
        **summary,
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
    }
    manifest_path = destination_dir / "manifest.json"
    manifest = {"version": 1, "milestones": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior = [
        item for item in manifest.get("milestones", [])
        if item.get("file") != destination.name
    ]
    manifest["milestones"] = sorted(
        prior + [record], key=lambda item: (item["step"], item["file"])
    )
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return destination
