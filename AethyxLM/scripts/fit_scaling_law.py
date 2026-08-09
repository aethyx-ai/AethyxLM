"""Fit a scaling curve from a JSON list of training-run summaries."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.scaling import fit_scaling_law


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, help="JSON list with parameters, tokens, val_loss")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = json.loads(args.runs.read_text(encoding="utf-8"))
    fit = fit_scaling_law(
        [run["parameters"] for run in runs],
        [run["tokens"] for run in runs],
        [run["val_loss"] for run in runs],
    )
    payload = asdict(fit)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
