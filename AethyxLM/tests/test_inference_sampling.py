import pytest
import torch

from inference.generation import (
    SamplingConfig,
    _apply_no_repeat_ngram,
    _apply_probability_filters,
    _apply_repetition_penalty,
    generate_batch_text,
    sampling_for_decoding,
)
from model.gpt import GPT


def test_sampling_config_rejects_invalid_ranges():
    with pytest.raises(ValueError):
        SamplingConfig(top_p=0)
    with pytest.raises(ValueError):
        SamplingConfig(repetition_penalty=0.9)
    with pytest.raises(ValueError):
        SamplingConfig(no_repeat_ngram_size=-1)


def test_named_sampled_profile_matches_interactive_defaults():
    assert sampling_for_decoding("sampled", max_new_tokens=77) == SamplingConfig(
        max_new_tokens=77
    )


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


class TinyTokenizer:
    bos_id = 1
    eos_id = 2
    pad_id = 0

    def encode(self, text):
        return [3 + (ord(character) % 12) for character in text]

    def decode(self, ids):
        return "".join(chr(97 + (token_id % 26)) for token_id in ids)


def test_batch_generation_preserves_input_order_for_unequal_prompt_lengths():
    torch.manual_seed(3)
    model = GPT(
        config={
            "vocab_size": 32,
            "context_length": 24,
            "embed_dim": 16,
            "num_heads": 4,
            "num_kv_heads": 2,
            "num_layers": 1,
            "ffn_dim": 32,
            "dropout": 0.0,
            "normalization": "rmsnorm",
            "position_encoding": "rope",
            "ffn_type": "swiglu",
            "fused_qkv": True,
        }
    ).eval()
    results = generate_batch_text(
        model,
        TinyTokenizer(),
        ("aa", "bbbb", "cc"),
        sampling=SamplingConfig(
            max_new_tokens=3,
            temperature=0,
            top_k=0,
            top_p=1,
            repetition_penalty=1,
            no_repeat_ngram_size=0,
        ),
    )

    assert len(results) == 3
    assert all(len(result.token_ids) <= 3 for result in results)
