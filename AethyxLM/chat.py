#!/usr/bin/env python3
"""Local completion and interactive chat interface for AethyxLM checkpoints."""

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer
from inference.generation import (
    SamplingConfig,
    format_inference_prompt as _format_inference_prompt,
    generate_text,
    sampling_for_decoding,
    stop_strings_for_mode,
)
from inference.prompt_contract import (
    PROMPT_CONTRACTS,
    PromptContract,
    resolve_prompt_contract,
    resolve_inference_mode,
    trim_serialized_history,
    critical_prefix_token_count,
)
from inference.retrieval import EvidenceIndex, EvidencePassage, format_evidence
from inference.quantization import QUANTIZATION_MODES, quantize_model_for_inference
from inference.tools import ToolController


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


def resolve_tokenizer_path(
    checkpoint_path: Path,
    checkpoint_config: dict,
    tokenizer_path: Optional[Path] = None,
) -> Path:
    """Resolve the checkpoint's tokenizer, preferring its saved fingerprint."""
    if tokenizer_path is not None:
        selected = tokenizer_path.expanduser().resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {selected}")
        return selected

    tokenizer_info = checkpoint_config.get("tokenizer", {})
    if not isinstance(tokenizer_info, dict):
        tokenizer_info = {}
    expected_hash = (
        tokenizer_info.get("sha256")
        or checkpoint_config.get("tokenizer_sha256")
    )
    saved_name = tokenizer_info.get("file_name")

    candidates = []
    if saved_name:
        candidates.extend(
            (
                checkpoint_path.parent / saved_name,
                PROJECT_ROOT / "tokenizer" / saved_name,
                PROJECT_ROOT / saved_name,
            )
        )
    candidates.append(PROJECT_ROOT / "tokenizer" / "tokenizer.json")
    candidates.extend((PROJECT_ROOT / "tokenizer").glob("*.json"))
    candidates.extend(checkpoint_path.parent.glob("*.json"))

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate not in seen and candidate.is_file():
            unique_candidates.append(candidate)
            seen.add(candidate)

    if expected_hash:
        for candidate in unique_candidates:
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest == expected_hash:
                return candidate
        expected = f" named {saved_name!r}" if saved_name else ""
        raise FileNotFoundError(
            f"Could not find the tokenizer{expected} matching checkpoint fingerprint "
            f"{expected_hash}. Pass it explicitly with --tokenizer."
        )

    if saved_name:
        named = next(
            (candidate for candidate in unique_candidates if candidate.name == saved_name),
            None,
        )
        if named is not None:
            return named
    if unique_candidates:
        return unique_candidates[0]
    raise FileNotFoundError(
        "No tokenizer JSON was found. Pass the correct file with --tokenizer."
    )


def load_model_and_tokenizer(
    checkpoint_path: Path,
    tokenizer_path: Optional[Path],
    device: str,
    quantization: str = "none",
):
    """Load and cross-check a checkpoint, its architecture, and tokenizer."""
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

    tokenizer_path = resolve_tokenizer_path(
        checkpoint_path,
        checkpoint_config,
        tokenizer_path,
    )
    print(f"Tokenizer: {tokenizer_path}")
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
    native_gqa_enabled = False
    if device.startswith("cuda"):
        for layer in model.layers:
            attention = layer.attention
            if attention.num_kv_heads != attention.num_heads and attention.use_sdpa:
                attention.native_gqa = "enable_gqa" in (
                    torch.nn.functional.scaled_dot_product_attention.__doc__ or ""
                )
                native_gqa_enabled |= attention.native_gqa
    model = quantize_model_for_inference(model, quantization, device)
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
    print(f"Quantization: {quantization}")
    print(f"Native GQA inference path: {'enabled with fallback' if native_gqa_enabled else 'disabled'}")
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
    stop_strings: tuple[str, ...] = (),
    on_text=None,
    return_result: bool = False,
    cache_strategy: str = "auto",
    preserve_prefix_tokens: int = 0,
):
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
        stop_strings=stop_strings,
        on_text=on_text,
        cache_strategy=cache_strategy,
        preserve_prefix_tokens=preserve_prefix_tokens,
    )
    return result if return_result else result.text


def format_inference_prompt(
    text: str,
    mode: str,
    history: str = "",
    prompt_contract: str = "legacy-chat-v1",
) -> str:
    """Compatibility wrapper for the canonical inference prompt formatter."""
    return _format_inference_prompt(text, mode, history, prompt_contract)


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


def interactive_session(
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
    mode: str,
    prompt_contract: PromptContract,
    tool_controller: ToolController | None = None,
    evidence_index: EvidenceIndex | None = None,
    retrieve_k: int = 4,
    cache_strategy: str = "auto",
    system_prompt: str | None = None,
):
    label = "Chat" if mode == "chat" else "Base completion"
    print(f"\n{label} mode started. Commands: /temp, /topk, /ngram, /max, /clear, /help, /quit")
    if tool_controller is not None:
        print(
            "Restricted Python tool enabled. Manual: "
            '/tool {"tool":"python","arguments":{"language":"python-restricted",'
            '"code":"print(sum(range(1, 11)))"}}'
        )
    if mode == "base":
        print("Each entry is continued as raw text; no User/Aethyx role markers are added.")
    else:
        print("Chat formatting is intended only for an instruction-tuned checkpoint.")
    history = (
        prompt_contract.format_system(system_prompt or "")
        if mode == "chat"
        else ""
    )

    while True:
        try:
            user_text = input("\nYou: " if mode == "chat" else "\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit", "q", "/quit"}:
            print("Bye!")
            return

        if user_text.startswith("/"):
            if user_text.startswith("/tool ") and tool_controller is not None:
                result = tool_controller.execute(user_text[len("/tool ") :])
                rendered = prompt_contract.format_tool_result(
                    str(result.get("tool", "invalid")), result
                )
                safe_print(rendered.strip())
                if mode == "chat":
                    history = trim_to_token_budget(
                        history + rendered,
                        tokenizer,
                        max(32, model.context_length - max_new - 32),
                    )
                continue
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

        prompt_text = user_text
        if evidence_index is not None:
            passages = evidence_index.search(user_text, limit=retrieve_k)
            evidence = format_evidence(
                passages, tokenizer, max_tokens=max(32, model.context_length // 3)
            )
            if evidence:
                prompt_text = (
                    "Use the evidence below and cite its SOURCE ID. Preserve identifiers "
                    f"and numbers exactly.\n\n{evidence}\n\nQuestion: {user_text}"
                )
                print("[Retrieved: " + ", ".join(item.source_id for item in passages) + "]")
        prompt = prompt_contract.format_user_turn(prompt_text, history)
        stop_strings = prompt_contract.stop_strings
        try:
            if stream and tool_controller is None:
                print("\nAethyx: ", end="", flush=True)
            result = generate(
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
                stop_strings=stop_strings,
                on_text=stream_write if stream and tool_controller is None else None,
                return_result=True,
                cache_strategy=cache_strategy,
                preserve_prefix_tokens=critical_prefix_token_count(
                    prompt, tokenizer, prompt_contract.name
                ),
            )
            response = result.text
            continuation_prompt = prompt
            if tool_controller is not None:
                visible_parts = []
                for tool_attempt in range(2):
                    emitted_call = tool_controller.execute_model_output(
                        response, malformed_attempt=tool_attempt
                    )
                    if emitted_call is None:
                        visible_parts.append(response)
                        break
                    if emitted_call["assistant_text"]:
                        visible_parts.append(emitted_call["assistant_text"])
                    tool_result = emitted_call["result"]
                    rendered = prompt_contract.format_tool_result(
                        str(tool_result.get("tool", "invalid")), tool_result
                    )
                    continuation_prompt = prompt_contract.continue_after_tool(
                        continuation_prompt, response, rendered
                    )
                    result = generate(
                        model,
                        tokenizer,
                        continuation_prompt,
                        max_new=max_new,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        stop_strings=stop_strings,
                        return_result=True,
                        cache_strategy=cache_strategy,
                        preserve_prefix_tokens=critical_prefix_token_count(
                            continuation_prompt, tokenizer, prompt_contract.name
                        ),
                    )
                    response = result.text
                else:
                    visible_parts.append(response.split("<TOOL_CALL>", 1)[0].rstrip())
                response = " ".join(part for part in visible_parts if part).strip()
            if stream:
                if tool_controller is not None:
                    safe_print(f"\nAethyx: {response}")
                else:
                    print()
            else:
                safe_print(f"\nAethyx: {response}")
            if mode == "chat":
                if result.dropped_prompt_tokens:
                    print(
                        f"[Context budget removed {result.dropped_prompt_tokens} older prompt tokens]"
                    )
                history_budget = max(32, model.context_length - max_new - 32)
                history, history_dropped = trim_serialized_history(
                    prompt_contract.append_assistant_turn(
                        continuation_prompt, response
                    ),
                    tokenizer,
                    history_budget,
                    prompt_contract.name,
                )
                if history_dropped:
                    print(f"[Removed {history_dropped} tokens in complete older turns]")
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
        default=None,
        help="Tokenizer JSON; by default it is detected from checkpoint metadata",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_MODES,
        default="none",
        help="Optional inference-only quantization (dynamic-int8 requires CPU)",
    )
    parser.add_argument(
        "--prompt",
        help="Run one completion with the selected mode and exit",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "base", "chat"),
        default="auto",
        help=(
            "auto uses checkpoint metadata; base performs raw continuation; chat "
            "requires a declared or explicitly selected prompt contract"
        ),
    )
    parser.add_argument(
        "--prompt-contract",
        choices=("auto", *PROMPT_CONTRACTS),
        default="auto",
        help="Prompt syntax; auto reads instruction-tuned checkpoint metadata",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.18)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=200)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--sampling-profile",
        choices=("custom", "prose", "code", "exact"),
        default="custom",
        help="Task-specific decoding defaults; custom uses the individual sampling flags",
    )
    parser.add_argument(
        "--enable-code-tool",
        action="store_true",
        help="Enable explicit restricted Python tool calls in interactive mode",
    )
    parser.add_argument(
        "--evidence-file",
        type=Path,
        action="append",
        default=[],
        help="UTF-8 evidence text file; may be repeated",
    )
    parser.add_argument("--retrieve-k", type=int, default=4)
    parser.add_argument("--system-prompt", help="Optional system instruction for chat mode")
    parser.add_argument(
        "--cache-strategy",
        choices=("auto", "preallocated", "tuple"),
        default="auto",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sampling_profile != "custom":
        profile = sampling_for_decoding(
            args.sampling_profile, max_new_tokens=args.max_new
        )
        args.temperature = profile.temperature
        args.top_k = profile.top_k
        args.top_p = profile.top_p
        args.min_p = profile.min_p
        args.repetition_penalty = profile.repetition_penalty
        args.no_repeat_ngram_size = profile.no_repeat_ngram_size
    print("=" * 60)
    print("AethyxLM - checkpoint interface")
    print("=" * 60)

    checkpoint_path = select_checkpoint(
        args.checkpoint,
        args.checkpoint_dir,
        use_latest=args.latest,
    )
    device = resolve_device(args.device)
    print(f"Device: {device}")
    model, tokenizer, checkpoint = load_model_and_tokenizer(
        checkpoint_path,
        args.tokenizer,
        device,
        quantization=args.quantization,
    )
    checkpoint_config = checkpoint.get("config", {})
    mode = resolve_inference_mode(args.mode, checkpoint_config)
    prompt_contract = resolve_prompt_contract(
        mode,
        checkpoint_config,
        None if args.prompt_contract == "auto" else args.prompt_contract,
    )
    evidence_index = None
    if args.evidence_file:
        passages = []
        for path in args.evidence_file:
            resolved = path.expanduser().resolve()
            passages.append(EvidencePassage(resolved.name, resolved.read_text(encoding="utf-8")))
        evidence_index = EvidenceIndex(passages)

    if args.prompt is not None:
        if args.stream:
            callback = stream_write
        else:
            callback = None
        one_shot_history = (
            prompt_contract.format_system(args.system_prompt or "")
            if mode == "chat"
            else ""
        )
        formatted_prompt = prompt_contract.format_user_turn(
            args.prompt, one_shot_history
        )
        result = generate(
            model,
            tokenizer,
            formatted_prompt,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            on_text=callback,
            stop_strings=prompt_contract.stop_strings,
            return_result=True,
            cache_strategy=args.cache_strategy,
            preserve_prefix_tokens=critical_prefix_token_count(
                formatted_prompt, tokenizer, prompt_contract.name
            ),
        )
        if args.stream:
            print()
        else:
            safe_print(result.text)
        if result.dropped_prompt_tokens:
            print(f"[Dropped {result.dropped_prompt_tokens} prompt tokens to fit context]")
        return

    interactive_session(
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
        mode=mode,
        prompt_contract=prompt_contract,
        tool_controller=(
            ToolController(allow_code=args.enable_code_tool)
            if args.enable_code_tool
            else None
        ),
        evidence_index=evidence_index,
        retrieve_k=max(1, args.retrieve_k),
        cache_strategy=args.cache_strategy,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
