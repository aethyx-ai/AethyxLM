"""Archive or inspect non-rotating training milestones."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.milestones import archive_milestone


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    archive = subparsers.add_parser("archive")
    archive.add_argument("checkpoint", type=Path)
    archive.add_argument("--milestone-dir", type=Path, default=Path("milestones"))
    archive.add_argument("--label")
    listing = subparsers.add_parser("list")
    listing.add_argument("--milestone-dir", type=Path, default=Path("milestones"))
    args = parser.parse_args()

    if args.command == "archive":
        destination = archive_milestone(
            args.checkpoint, args.milestone_dir, label=args.label
        )
        print(destination)
        return

    manifest = args.milestone_dir / "manifest.json"
    if not manifest.exists():
        print("No milestone manifest found")
        return
    print(json.dumps(json.loads(manifest.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
