"""KV-cached generation with modern, testable sampling controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import torch


InferenceMode = Literal["base", "chat"]
CHAT_STOP_STRINGS = ("\nUser:", "User:", "\nAethyx:", "Aethyx:")


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    min_p: float = 0.0
    repetition_penalty: float = 1.18
    repetition_window: int = 128
    no_repeat_ngram_size: int = 4

    def __post_init__(self):
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be within (0, 1]")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be within [0, 1]")
        if self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be at least 1")
        if self.repetition_window < 0:
            raise ValueError("repetition_window cannot be negative")
        if self.no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size cannot be negative")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str


def format_inference_prompt(
    text: str,
    mode: InferenceMode,
    history: str = "",
) -> str:
    """Apply the one canonical prompt contract used by chat and evaluation."""
    if mode == "base":
        return text
    if mode == "chat":
        return f"{history}User: {text}\nAethyx:"
    raise ValueError(f"Unsupported inference mode: {mode}")


def stop_strings_for_mode(mode: InferenceMode) -> tuple[str, ...]:
    """Return the canonical decoding stops for an inference mode."""
    if mode == "base":
        return ()
    if mode == "chat":
        return CHAT_STOP_STRINGS
    raise ValueError(f"Unsupported inference mode: {mode}")


def sampling_for_decoding(
    decoding: Literal["default", "greedy", "sampled"] = "default",
    *,
    max_new_tokens: int = 200,
) -> SamplingConfig:
    """Create named sampling profiles so CLIs and evaluations cannot drift."""
    if decoding == "default":
        return SamplingConfig(max_new_tokens=max_new_tokens)
    if decoding == "greedy":
        return SamplingConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )
    if decoding == "sampled":
        return SamplingConfig(max_new_tokens=max_new_tokens)
    raise ValueError(f"Unsupported decoding profile: {decoding}")


def _apply_repetition_penalty(
    logits: torch.Tensor,
    recent_ids: Sequence[int],
    penalty: float,
):
    if penalty == 1 or not recent_ids:
        return logits
    unique_ids = torch.tensor(
        sorted(set(int(value) for value in recent_ids)),
        device=logits.device,
        dtype=torch.long,
    )
    selected = logits[:, unique_ids]
    adjusted = torch.where(selected < 0, selected * penalty, selected / penalty)
    logits[:, unique_ids] = adjusted
    return logits


def _apply_no_repeat_ngram(
    logits: torch.Tensor,
    token_ids: Sequence[int],
    ngram_size: int,
):
    """Block tokens that would recreate an n-gram already in the sequence."""
    if ngram_size <= 0 or len(token_ids) + 1 < ngram_size:
        return logits

    if ngram_size == 1:
        blocked = {int(token_id) for token_id in token_ids}
    else:
        prefix = tuple(int(value) for value in token_ids[-(ngram_size - 1) :])
        blocked = {
            int(token_ids[index + ngram_size - 1])
            for index in range(len(token_ids) - ngram_size + 1)
            if tuple(
                int(value)
                for value in token_ids[index : index + ngram_size - 1]
            )
            == prefix
        }

    valid = sorted(token_id for token_id in blocked if 0 <= token_id < logits.size(-1))
    if not valid:
        return logits

    original = logits.clone()
    logits[:, valid] = -float("inf")
    return logits if torch.isfinite(logits).any(dim=-1).all() else original


def _apply_probability_filters(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    min_p: float,
):
    if top_k:
        threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < threshold, -float("inf"))

    if top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(
            -1, sorted_indices, sorted_logits
        )

    if min_p > 0:
        probabilities = torch.softmax(logits, dim=-1)
        threshold = probabilities.max(dim=-1, keepdim=True).values * min_p
        logits = logits.masked_fill(probabilities < threshold, -float("inf"))
    return logits


def _truncate_stop_strings(text: str, stop_strings: Sequence[str]):
    lowered = text.lower()
    positions = [
        lowered.find(value.lower()) for value in stop_strings if value
    ]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return text, False
    return text[: min(positions)].rstrip(), True


def _sample_token(
    logits: torch.Tensor,
    token_ids: Sequence[int],
    sampling: SamplingConfig,
) -> torch.Tensor:
    logits = _apply_repetition_penalty(
        logits,
        token_ids[-sampling.repetition_window :],
        sampling.repetition_penalty,
    )
    logits = _apply_no_repeat_ngram(
        logits,
        token_ids,
        sampling.no_repeat_ngram_size,
    )
    if sampling.temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = _apply_probability_filters(
        logits / sampling.temperature,
        sampling.top_k,
        sampling.top_p,
        sampling.min_p,
    )
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    sampling: SamplingConfig | None = None,
    stop_strings: Sequence[str] = (),
    on_text: Optional[Callable[[str], None]] = None,
) -> GenerationResult:
    """Generate one continuation and optionally stream decoded text deltas."""
    sampling = sampling or SamplingConfig()
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        if tokenizer.bos_id is None:
            raise ValueError("prompt produced no tokens and tokenizer has no BOS token")
        prompt_ids = [tokenizer.bos_id]

    device = next(model.parameters()).device
    context_length = int(model.context_length)
    sequence = torch.tensor(
        [prompt_ids[-context_length:]], dtype=torch.long, device=device
    )
    generated: list[int] = []
    emitted_text = ""
    finish_reason = "length"
    model.eval()
    logits, cache = model(sequence, use_cache=True)

    for _ in range(sampling.max_new_tokens):
        next_logits = logits[:, -1, :].float()
        for blocked in (tokenizer.pad_id, tokenizer.bos_id):
            if blocked is not None:
                next_logits[:, blocked] = -float("inf")

        next_id = _sample_token(next_logits, prompt_ids + generated, sampling)

        token_id = int(next_id.item())
        if tokenizer.eos_id is not None and token_id == tokenizer.eos_id:
            finish_reason = "eos"
            break
        generated.append(token_id)
        sequence = torch.cat((sequence, next_id), dim=1)

        decoded = tokenizer.decode(generated)
        visible, stopped = _truncate_stop_strings(decoded, stop_strings)
        if on_text is not None and visible.startswith(emitted_text):
            delta = visible[len(emitted_text) :]
            if delta:
                on_text(delta)
            emitted_text = visible
        if stopped:
            finish_reason = "stop"
            break

        cached_length = int(cache[0][0].size(2))
        if cached_length >= context_length:
            logits, cache = model(sequence[:, -context_length:], use_cache=True)
        else:
            logits, cache = model(next_id, kv_cache=cache, use_cache=True)

    text, _ = _truncate_stop_strings(tokenizer.decode(generated), stop_strings)
    return GenerationResult(text=text.strip(), token_ids=tuple(generated), finish_reason=finish_reason)


@torch.inference_mode()
def generate_batch_text(
    model,
    tokenizer,
    prompts: Sequence[str],
    sampling: SamplingConfig | None = None,
    stop_strings: Sequence[str] = (),
) -> list[GenerationResult]:
    """Generate continuations in real batches, bucketing unequal prompt lengths."""
    sampling = sampling or SamplingConfig()
    if not prompts:
        return []
    context_length = int(model.context_length)
    encoded = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        if not ids:
            if tokenizer.bos_id is None:
                raise ValueError("prompt produced no tokens and tokenizer has no BOS token")
            ids = [tokenizer.bos_id]
        encoded.append(ids[-context_length:])

    buckets: dict[int, list[int]] = {}
    for index, ids in enumerate(encoded):
        buckets.setdefault(len(ids), []).append(index)
    results: list[GenerationResult | None] = [None] * len(prompts)
    device = next(model.parameters()).device
    model.eval()

    for indices in buckets.values():
        prompt_rows = [encoded[index] for index in indices]
        sequence = torch.tensor(prompt_rows, dtype=torch.long, device=device)
        generated = [[] for _ in indices]
        finish_reasons = ["length" for _ in indices]
        finished = [False for _ in indices]
        logits, cache = model(sequence, use_cache=True)

        for _ in range(sampling.max_new_tokens):
            next_values = []
            for row, prompt_ids in enumerate(prompt_rows):
                if finished[row]:
                    fallback = tokenizer.eos_id
                    if fallback is None:
                        fallback = tokenizer.pad_id if tokenizer.pad_id is not None else 0
                    next_values.append(int(fallback))
                    continue
                row_logits = logits[row : row + 1, -1, :].float()
                for blocked in (tokenizer.pad_id, tokenizer.bos_id):
                    if blocked is not None:
                        row_logits[:, blocked] = -float("inf")
                next_id = _sample_token(
                    row_logits, prompt_ids + generated[row], sampling
                )
                token_id = int(next_id.item())
                next_values.append(token_id)
                if tokenizer.eos_id is not None and token_id == tokenizer.eos_id:
                    finish_reasons[row] = "eos"
                    finished[row] = True
                    continue
                generated[row].append(token_id)
                _, stopped = _truncate_stop_strings(
                    tokenizer.decode(generated[row]), stop_strings
                )
                if stopped:
                    finish_reasons[row] = "stop"
                    finished[row] = True

            if all(finished):
                break
            next_tensor = torch.tensor(next_values, device=device)[:, None]
            sequence = torch.cat((sequence, next_tensor), dim=1)
            cached_length = int(cache[0][0].size(2))
            if cached_length >= context_length:
                logits, cache = model(sequence[:, -context_length:], use_cache=True)
            else:
                logits, cache = model(next_tensor, kv_cache=cache, use_cache=True)

        for row, original_index in enumerate(indices):
            text, _ = _truncate_stop_strings(
                tokenizer.decode(generated[row]), stop_strings
            )
            results[original_index] = GenerationResult(
                text=text.strip(),
                token_ids=tuple(generated[row]),
                finish_reason=finish_reasons[row],
            )
    return [result for result in results if result is not None]
