"""Checkpoint-wide multilingual, generation, and long-context evaluation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from dataset.dataset import AethyxDataset
from evaluation.evaluator import evaluate_language_model
from evaluation.long_context import evaluate_passkey_retrieval
from inference.generation import SamplingConfig, generate_text
from model.gpt import GPT


DEFAULT_PROMPTS = (
    "India is a diverse country because",
    "भारत में शिक्षा का महत्व",
    "বাংলা ভাষার ইতিহাস",
    "Solve step by step: 17 × 23 =",
)


def _repetition_metrics(token_ids: Iterable[int]) -> dict:
    ids = list(token_ids)
    if not ids:
        return {"tokens": 0, "unique_token_ratio": 0.0, "repeated_4gram_ratio": 0.0}
    ngrams = [tuple(ids[index : index + 4]) for index in range(max(0, len(ids) - 3))]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "tokens": len(ids),
        "unique_token_ratio": len(set(ids)) / len(ids),
        "repeated_4gram_ratio": repeated / max(len(ngrams), 1),
    }


def load_checkpoint_once(checkpoint_path: Path, tokenizer, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    expected_hash = (
        config.get("tokenizer", {}).get("sha256")
        or config.get("tokenizer_sha256")
    )
    if expected_hash and expected_hash != tokenizer.sha256:
        raise RuntimeError(f"Tokenizer mismatch for {checkpoint_path}")
    state = checkpoint["model_state_dict"]
    model_config = GPT._infer_checkpoint_config(
        state, config.get("model", config)
    )
    if int(model_config["vocab_size"]) != tokenizer.vocab_size:
        raise RuntimeError(f"Vocabulary mismatch for {checkpoint_path}")
    model = GPT(vocab_size=tokenizer.vocab_size, config=model_config)
    model.load_compatible_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint


def evaluate_registry(
    model,
    tokenizer,
    registry: dict,
    project_root: Path,
    batch_size: int,
    max_batches: int,
    device: str,
):
    results = {}
    for name, entry in registry.items():
        validation_path = project_root / entry["val"]
        dataset = AethyxDataset(
            validation_path,
            context_length=model.context_length,
            tokenizer_path=tokenizer.path,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        metrics = evaluate_language_model(
            model, loader, device=device, max_batches=max_batches
        )
        results[name] = {**asdict(metrics), "weight": float(entry["weight"])}
    weighted_loss = sum(
        item["loss"] * item["weight"] for item in results.values()
    )
    return {
        "per_source": results,
        "weighted_loss": weighted_loss,
        "weighted_perplexity": math.exp(min(weighted_loss, 20.0)),
    }


def evaluate_generation(model, tokenizer, prompts=DEFAULT_PROMPTS, max_new_tokens=64):
    sampling = SamplingConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.05,
    )
    results = []
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, sampling=sampling)
        results.append(
            {
                "prompt": prompt,
                "continuation": generated.text,
                "finish_reason": generated.finish_reason,
                **_repetition_metrics(generated.token_ids),
            }
        )
    return results


def evaluate_checkpoint(
    checkpoint_path: Path,
    tokenizer,
    registry: dict,
    project_root: Path,
    device: str,
    batch_size: int = 4,
    max_batches: int = 20,
    context_lengths=(128, 256, 512),
    long_context_trials: int = 2,
    generation_tokens: int = 64,
):
    model, checkpoint = load_checkpoint_once(checkpoint_path, tokenizer, device)
    valid_lengths = [
        int(length) for length in context_lengths if int(length) <= model.context_length
    ]
    payload = {
        "path": str(checkpoint_path.resolve()),
        "step": int(checkpoint.get("step", -1)),
        "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        "best_val_loss": (
            float(checkpoint["best_val_loss"])
            if math.isfinite(float(checkpoint.get("best_val_loss", float("inf"))))
            else None
        ),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "tokenizer_sha256": tokenizer.sha256,
        "language_model": evaluate_registry(
            model,
            tokenizer,
            registry,
            project_root,
            batch_size,
            max_batches,
            device,
        ),
        "generation_diagnostics": evaluate_generation(
            model, tokenizer, max_new_tokens=generation_tokens
        ),
        "long_context": [
            {**asdict(result), "accuracy": result.accuracy}
            for result in evaluate_passkey_retrieval(
                model,
                tokenizer,
                valid_lengths,
                trials=long_context_trials,
            )
        ],
    }
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload
