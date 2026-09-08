#!/usr/bin/env python3
"""Evaluate a checkpoint through generated answers, never multiple choice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chat import load_model_and_tokenizer, resolve_device
from evaluation.freeform_suite import evaluate_freeform_cases
from inference.prompt_contract import PROMPT_CONTRACTS, resolve_inference_mode, resolve_prompt_contract


def parse_seeds(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(item.strip()) for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--prompt-mode", choices=("auto", "base", "chat", "both"), default="auto")
    parser.add_argument(
        "--prompt-contract", choices=("auto", *PROMPT_CONTRACTS), default="auto"
    )
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, tokenizer, checkpoint = load_model_and_tokenizer(
        args.checkpoint.resolve(), args.tokenizer, device
    )
    checkpoint_config = checkpoint.get("config", {})
    inferred_mode = resolve_inference_mode("auto", checkpoint_config)
    modes = (
        ("base", "chat")
        if args.prompt_mode == "both"
        else (inferred_mode,)
        if args.prompt_mode == "auto"
        else (args.prompt_mode,)
    )
    contracts = {
        mode: resolve_prompt_contract(
            mode,
            checkpoint_config,
            None if args.prompt_contract == "auto" else args.prompt_contract,
        )
        for mode in modes
    }
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "step": checkpoint.get("step"),
        "device": device,
        "modes": {
            mode: evaluate_freeform_cases(
                model,
                tokenizer,
                prompt_mode=mode,
                max_new_tokens=args.max_new,
                seeds=parse_seeds(args.seeds),
                prompt_contract=contracts[mode].name,
            )
            for mode in modes
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({mode: report["greedy"] for mode, report in result["modes"].items()}, indent=2))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
