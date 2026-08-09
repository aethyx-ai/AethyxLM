"""Reproducible language-model and context-efficiency measurements."""

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import torch

@dataclass(frozen=True)
class LanguageModelMetrics:
    loss: float
    perplexity: float
    tokens: int
    batches: int


@dataclass(frozen=True)
class ContextCompressionMetrics:
    source_units: int
    compressed_units: int
    reduction: float
    baseline_accuracy: float
    compressed_accuracy: float
    accuracy_retention: float


@torch.no_grad()
def evaluate_language_model(
    model,
    dataloader: Iterable,
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> LanguageModelMetrics:
    """Measure token-weighted loss instead of averaging uneven batch means."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    for input_ids, targets in dataloader:
        if max_batches is not None and batches >= max_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        logits = model(input_ids)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        total_loss += torch.nn.functional.cross_entropy(
            flat_logits, flat_targets, reduction="sum", ignore_index=-100
        ).item()
        total_tokens += int((flat_targets != -100).sum().item())
        batches += 1
    if was_training:
        model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    return LanguageModelMetrics(
        loss=mean_loss,
        perplexity=math.exp(min(mean_loss, 20.0)),
        tokens=total_tokens,
        batches=batches,
    )


def context_compression_metrics(
    source_units: int,
    compressed_units: int,
    baseline_correct: int,
    compressed_correct: int,
    examples: int,
) -> ContextCompressionMetrics:
    """Score compression only alongside retained downstream accuracy."""
    if source_units <= 0 or compressed_units <= 0 or examples <= 0:
        raise ValueError("unit counts and examples must be positive")
    baseline_accuracy = baseline_correct / examples
    compressed_accuracy = compressed_correct / examples
    retention = (
        compressed_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0.0
    )
    return ContextCompressionMetrics(
        source_units=source_units,
        compressed_units=compressed_units,
        reduction=1.0 - compressed_units / source_units,
        baseline_accuracy=baseline_accuracy,
        compressed_accuracy=compressed_accuracy,
        accuracy_retention=retention,
    )


def metrics_dict(metrics):
    return asdict(metrics)
