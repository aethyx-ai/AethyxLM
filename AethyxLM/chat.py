#!/usr/bin/env python3
"""Local completion and interactive chat interface for AethyxLM checkpoints."""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer
from inference.generation import SamplingConfig, generate_text


def checkpoint_step(path: Path) -> Optional[int]:
    """Return the numbered checkpoint step, when present in the filename."""
    match = re.fullmatch(r"checkpoint_step_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def discover_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """List non-empty checkpoint files in a predictable, useful order."""
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    candidates = [
        path for path in checkpoint_dir.glob("*.pt")
        if path.is_file() and path.stat().st_size > 0
    ]

    def sort_key(path: Path):
        if path.name == "checkpoint_latest.pt":
            return (0, 0, -path.stat().st_mtime)
        step = checkpoint_step(path)
        if step is not None:
            return (1, -step, -path.stat().st_mtime)
        if path.name == "checkpoint_best.pt":
            return (2, 0, -path.stat().st_mtime)
        return (3, 0, -path.stat().st_mtime)

    return sorted(candidates, key=sort_key)


def newest_checkpoint(candidates: list[Path]) -> Path:
    """Prefer the trainer's latest alias, then the highest numbered checkpoint."""
    if not candidates:
        raise FileNotFoundError("No checkpoint files were found")

    latest_alias = next(
        (path for path in candidates if path.name == "checkpoint_latest.pt"), None
    )
    if latest_alias is not None:
        return latest_alias

    numbered = [path for path in candidates if checkpoint_step(path) is not None]
    if numbered:
        return max(numbered, key=lambda path: checkpoint_step(path) or -1)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def select_checkpoint(
    checkpoint: Optional[Path],
    checkpoint_dir: Path,
    use_latest: bool = False,
) -> Path:
    """Resolve an explicit checkpoint or offer an interactive selection menu."""
    if checkpoint is not None:
        selected = checkpoint.expanduser().resolve()
        if not selected.is_file() or selected.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint not found or empty: {selected}")
        return selected

    candidates = discover_checkpoints(checkpoint_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No .pt checkpoints found in {checkpoint_dir.expanduser().resolve()}"
        )

    if use_latest or len(candidates) == 1 or not sys.stdin.isatty():
        return newest_checkpoint(candidates)

    print("\nAvailable checkpoints:")
    for index, path in enumerate(candidates, start=1):
        step = checkpoint_step(path)
        step_text = f"step {step:,}" if step is not None else path.stem
        size_mib = path.stat().st_size / (1024 * 1024)
        print(f"  [{index}] {path.name}  ({step_text}, {size_mib:.1f} MiB)")

    while True:
        choice = input(f"Select checkpoint [1-{len(candidates)}] (default 1): ").strip()
        if not choice:
            return candidates[0]
        try:
            selected_index = int(choice) - 1
        except ValueError:
            selected_index = -1
        if 0 <= selected_index < len(candidates):
            return candidates[selected_index]
        print("Invalid selection.")


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return requested


def load_model_and_tokenizer(
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: str,
):
    """Load and cross-check a checkpoint, its architecture, and tokenizer v2."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("Expected a training checkpoint containing model_state_dict")

    state_dict = checkpoint["model_state_dict"]
    checkpoint_config = checkpoint.get("config", {})
    saved_model_config = checkpoint_config.get("model", checkpoint_config)
    model_config = GPT._infer_checkpoint_config(state_dict, saved_model_config)

    tokenizer_path = tokenizer_path.expanduser().resolve()
    tokenizer = AethyxTokenizer(tokenizer_path)
    expected_vocab = int(model_config["vocab_size"])
    if tokenizer.vocab_size != expected_vocab:
        raise RuntimeError(
            f"Tokenizer vocabulary is {tokenizer.vocab_size:,}, but the checkpoint "
            f"expects {expected_vocab:,}. Use the tokenizer that produced this checkpoint."
        )

    tokenizer_info = checkpoint_config.get("tokenizer", {})
    expected_hash = (
        tokenizer_info.get("sha256")
        if isinstance(tokenizer_info, dict)
        else None
    ) or checkpoint_config.get("tokenizer_sha256")
    if expected_hash and tokenizer.sha256 != expected_hash:
        raise RuntimeError(
            "Tokenizer fingerprint mismatch.\n"
            f"Checkpoint: {expected_hash}\n"
            f"Selected:   {tokenizer.sha256}\n"
            f"Path:       {tokenizer_path}"
        )

    model = GPT(vocab_size=expected_vocab, config=model_config)
    model.load_compatible_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(f"Loaded training step: {checkpoint.get('step', 'unknown')}")
    print(f"Architecture: {model.num_layers} layers, {model.embed_dim} dimensions")
    tokenizer_check = (
        "fingerprint verified"
        if expected_hash
        else "vocabulary verified; checkpoint has no tokenizer fingerprint"
    )
    print(f"Vocabulary: {tokenizer.vocab_size:,} ({tokenizer_check})")
    print(f"Context length: {model.context_length:,}")
    return model, tokenizer, checkpoint


def truncate_at_turn_marker(text: str) -> str:
    """Discard a generated next-speaker marker without rewriting valid text."""
    lowered = text.lower()
    positions = [
        lowered.find(marker)
        for marker in ("\nuser:", "user:", "\naethyx:", "aethyx:")
    ]
    positions = [position for position in positions if position >= 0]
    return text[:min(positions)].rstrip() if positions else text.strip()


def generate(
    model: GPT,
    tokenizer: AethyxTokenizer,
    prompt: str,
    max_new: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    min_p: float = 0.0,
    repetition_penalty: float = 1.18,
    no_repeat_ngram_size: int = 4,
    on_text=None,
) -> str:
    """Compatibility wrapper around the reusable inference engine."""
    result = generate_text(
        model,
        tokenizer,
        prompt,
        sampling=SamplingConfig(
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        ),
        stop_strings=("\nUser:", "User:", "\nAethyx:", "Aethyx:"),
        on_text=on_text,
    )
    return result.text


def trim_to_token_budget(
    text: str,
    tokenizer: AethyxTokenizer,
    max_tokens: int,
) -> str:
    """Keep the newest complete token budget for conversational history."""
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[-max_tokens:])


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def stream_write(text: str):
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.write(text.encode("ascii", "replace").decode("ascii"))
    sys.stdout.flush()


def interactive_chat(
    model: GPT,
    tokenizer: AethyxTokenizer,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    max_new: int,
    stream: bool,
):
    print("\nChat started. Commands: /temp, /topk, /ngram, /max, /clear, /help, /quit")
    print(
        "Note: this is a pretrained base-model checkpoint; conversational quality "
        "depends on later instruction tuning."
    )
    history = ""

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit", "q", "/quit"}:
            print("Bye!")
            return

        if user_text.startswith("/"):
            parts = user_text.split(maxsplit=1)
            command = parts[0].lower()
            value = parts[1] if len(parts) == 2 else None
            try:
                if command == "/temp" and value is not None:
                    temperature = float(value)
                    if temperature <= 0:
                        raise ValueError
                    print(f"[Temperature = {temperature}]")
                elif command == "/topk" and value is not None:
                    top_k = int(value)
                    if top_k < 0:
                        raise ValueError
                    print(f"[Top-k = {top_k}]")
                elif command == "/ngram" and value is not None:
                    no_repeat_ngram_size = int(value)
                    if no_repeat_ngram_size < 0:
                        raise ValueError
                    print(f"[No-repeat n-gram size = {no_repeat_ngram_size}]")
                elif command == "/max" and value is not None:
                    max_new = int(value)
                    if max_new <= 0:
                        raise ValueError
                    print(f"[Maximum new tokens = {max_new}]")
                elif command in {"/clear", "/reset"}:
                    history = ""
                    print("[Context cleared]")
                elif command == "/help":
                    print(
                        "/temp <number>, /topk <integer>, /ngram <integer>, "
                        "/max <integer>, /clear, /quit"
                    )
                else:
                    print("Unknown or incomplete command. Type /help.")
            except ValueError:
                print("Invalid command value. Type /help.")
            continue

        prompt = f"{history}User: {user_text}\nAethyx:"
        try:
            if stream:
                print("\nAethyx: ", end="", flush=True)
            response = generate(
                model,
                tokenizer,
                prompt,
                max_new=max_new,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                on_text=stream_write if stream else None,
            )
            if stream:
                print()
            else:
                safe_print(f"\nAethyx: {response}")
            history_budget = max(32, model.context_length - max_new - 32)
            history = trim_to_token_budget(
                f"{prompt} {response}\n",
                tokenizer,
                history_budget,
            )
        except Exception as error:
            print(f"[Generation error] {error}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Exact checkpoint file to load (for example checkpoint_step_21000.pt)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
        help="Directory shown by the interactive checkpoint selector",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Load checkpoint_latest.pt without showing the selection menu",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=PROJECT_ROOT / "tokenizer" / "tokenizer.json",
        help="Tokenizer JSON; defaults to the active tokenizer v2",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--prompt",
        help="Run one raw completion and exit instead of opening interactive chat",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.18)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=200)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("AethyxLM - tokenizer v2 checkpoint interface")
    print("=" * 60)

    checkpoint_path = select_checkpoint(
        args.checkpoint,
        args.checkpoint_dir,
        use_latest=args.latest,
    )
    device = resolve_device(args.device)
    print(f"Device: {device}")
    model, tokenizer, _ = load_model_and_tokenizer(
        checkpoint_path,
        args.tokenizer,
        device,
    )

    if args.prompt is not None:
        if args.stream:
            callback = stream_write
        else:
            callback = None
        continuation = generate(
            model,
            tokenizer,
            args.prompt,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            on_text=callback,
        )
        if args.stream:
            print()
        else:
            safe_print(continuation)
        return

    interactive_chat(
        model,
        tokenizer,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_new=args.max_new,
        stream=args.stream,
    )


if __name__ == "__main__":
    main()
