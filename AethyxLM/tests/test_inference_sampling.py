import pytest
import torch

from inference.generation import (
    SamplingConfig,
    _apply_probability_filters,
    _apply_repetition_penalty,
)


def test_sampling_config_rejects_invalid_ranges():
    with pytest.raises(ValueError):
        SamplingConfig(top_p=0)
    with pytest.raises(ValueError):
        SamplingConfig(repetition_penalty=0.9)


def test_sampling_filters_and_repetition_penalty_change_expected_logits():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    penalized = _apply_repetition_penalty(logits.clone(), [0], 2.0)
    assert penalized[0, 0] == 2.0
    filtered = _apply_probability_filters(logits.clone(), top_k=2, top_p=1.0, min_p=0)
    assert torch.isfinite(filtered[0, :2]).all()
    assert torch.isneginf(filtered[0, 2:]).all()

