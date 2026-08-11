"""Compile or benchmark local context representations."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from context_lab.benchmark import BenchmarkCase, benchmark_compiler
from context_lab.compiler import CompilerPolicy, LocalContextCompiler
from context_lab.schema import ContextItem, ContextRequest
from context_lab.visual import render_visual_pages
from tokenizer.tokenizer import AethyxTokenizer


def request_from_dict(payload):
    return ContextRequest(
        query=payload["query"],
        request_id=payload.get("request_id", ""),
        items=tuple(ContextItem(**item) for item in payload["items"]),
    )


def compiler_from_args(args, tokenizer=None):
    return LocalContextCompiler(
        CompilerPolicy(
            mode=args.mode,
            max_selected_items=args.max_selected_items,
            target_model_supports_vision=args.target_model_supports_vision,
        ),
        token_counter=(tokenizer.encode if tokenizer is not None else None),
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("compile", "benchmark"):
        sub = subparsers.add_parser(name)
        sub.add_argument("input", type=Path)
        sub.add_argument("--mode", choices=("graph", "visual", "hybrid"), default="graph")
        sub.add_argument("--max-selected-items", type=int, default=12)
        sub.add_argument("--target-model-supports-vision", action="store_true")
        sub.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer/tokenizer.json")
        sub.add_argument("--output", type=Path, required=True)
    subparsers.choices["compile"].add_argument("--render-dir", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    tokenizer = AethyxTokenizer(args.tokenizer)
    compiler = compiler_from_args(args, tokenizer)
    if args.command == "compile":
        compiled = compiler.compile(request_from_dict(payload))
        result = compiled.manifest()
        if args.render_dir:
            rendered = render_visual_pages(compiled.visual_pages, args.render_dir)
            result["rendered_pages"] = [str(path.resolve()) for path in rendered]
    else:
        cases = [
            BenchmarkCase(
                request=request_from_dict(case),
                relevant_source_ids=tuple(case["relevant_source_ids"]),
                exact_values=tuple(case.get("exact_values", ())),
            )
            for case in payload["cases"]
        ]
        result = benchmark_compiler(compiler, tokenizer, cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
