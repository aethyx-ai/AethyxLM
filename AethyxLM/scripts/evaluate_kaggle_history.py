"""Resumably download and evaluate versioned Kaggle checkpoints one at a time."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import kagglehub


ROOT = Path(__file__).resolve().parents[1]
DATASET = "aethyx/aethyxlm-v3-live-checkpoints"


def parse_versions(value: str) -> list[int]:
    if value == "all":
        return list(range(1, 112))
    versions = sorted({int(item) for item in value.split(",") if item.strip()})
    if not versions or any(version < 1 or version > 111 for version in versions):
        raise argparse.ArgumentTypeError("versions must be comma-separated values from 1 to 111, or 'all'")
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--versions",
        type=parse_versions,
        default=parse_versions("10,20,30,40,50,60,70,80,90,100,110,111"),
        help="Kaggle dataset versions, or 'all' (default: 10K milestones plus 111K)",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/results/kaggle_history")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}

    with tempfile.TemporaryDirectory(prefix="aethyx_kaggle_eval_") as temporary:
        download_dir = Path(temporary)
        for version in args.versions:
            step = version * 1000
            output = args.output_dir / f"v3_step_{step}_deep.json"
            if output.exists():
                status[str(version)] = {"step": step, "status": "already_complete", "output": str(output)}
                status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                continue

            filename = f"checkpoint_step_{step}.pt"
            handle = f"{DATASET}/versions/{version}"
            checkpoint = Path(
                kagglehub.dataset_download(
                    handle,
                    path=filename,
                    force_download=True,
                    output_dir=str(download_dir),
                )
            )
            command = [
                sys.executable,
                str(ROOT / "scripts/evaluate_checkpoint_deep.py"),
                str(checkpoint),
                "--output", str(output),
                "--device", "cpu",
                "--threads", str(max(1, args.threads)),
                "--validation-batches", "1",
                "--context-cases", "5",
                "--generation-tokens", "16",
                "--passkey-lengths", "128,512,1024",
            ]
            status[str(version)] = {"step": step, "status": "running"}
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            subprocess.run(command, cwd=ROOT, check=True)
            status[str(version)] = {"step": step, "status": "complete", "output": str(output)}
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
