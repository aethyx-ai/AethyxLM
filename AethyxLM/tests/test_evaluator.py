import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.evaluator import context_compression_metrics, evaluate_language_model
from model.gpt import GPT


def test_language_model_evaluation_reports_tokens_and_perplexity():
    model = GPT(
        config={
            "vocab_size": 32,
            "context_length": 8,
            "embed_dim": 16,
            "num_heads": 4,
            "num_layers": 1,
            "ffn_dim": 32,
            "dropout": 0.0,
        }
    )
    values = torch.randint(0, 32, (4, 8))
    loader = DataLoader(TensorDataset(values, values), batch_size=2)
    metrics = evaluate_language_model(model, loader, max_batches=1)
    assert metrics.tokens == 16
    assert metrics.batches == 1
    assert metrics.perplexity == pytest.approx(math.exp(metrics.loss))


def test_compression_score_keeps_accuracy_and_reduction_separate():
    metrics = context_compression_metrics(1000, 300, 90, 87, 100)
    assert metrics.reduction == pytest.approx(0.7)
    assert metrics.baseline_accuracy == pytest.approx(0.9)
    assert metrics.compressed_accuracy == pytest.approx(0.87)
    assert metrics.accuracy_retention == pytest.approx(0.87 / 0.9)
