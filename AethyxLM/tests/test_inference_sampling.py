import pytest
import torch

from inference.generation import (
    SamplingConfig,
    _apply_no_repeat_ngram,
    _apply_probability_filters,
    _apply_repetition_penalty,
)


def test_sampling_config_rejects_invalid_ranges():
    with pytest.raises(ValueError):
        SamplingConfig(top_p=0)
    with pytest.raises(ValueError):
        SamplingConfig(repetition_penalty=0.9)
    with pytest.raises(ValueError):
        SamplingConfig(no_repeat_ngram_size=-1)


def test_sampling_filters_and_repetition_penalty_change_expected_logits():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    penalized = _apply_repetition_penalty(logits.clone(), [0], 2.0)
    assert penalized[0, 0] == 2.0
    filtered = _apply_probability_filters(logits.clone(), top_k=2, top_p=1.0, min_p=0)
    assert torch.isfinite(filtered[0, :2]).all()
    assert torch.isneginf(filtered[0, 2:]).all()


def test_no_repeat_ngram_blocks_only_the_repeated_continuation():
    logits = torch.zeros(1, 10)
    sequence = [1, 2, 3, 4, 2, 3]
    blocked = _apply_no_repeat_ngram(logits, sequence, ngram_size=3)

    assert torch.isneginf(blocked[0, 4])
    assert torch.isfinite(blocked[0, 5])


def test_no_repeat_ngram_can_be_disabled():
    logits = torch.zeros(1, 4)
    unchanged = _apply_no_repeat_ngram(logits, [1, 1], ngram_size=0)
    assert torch.equal(unchanged, logits)
