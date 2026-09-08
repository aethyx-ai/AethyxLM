import pytest
import torch

from inference.generation import SamplingConfig, generate_text, sampling_for_decoding
from inference.prompt_contract import (
    critical_prefix_token_count,
    resolve_inference_mode,
    resolve_prompt_contract,
)
from model.attention import PreallocatedKVCache
from model.gpt import GPT


def tiny_model(context_length=12):
    torch.manual_seed(7)
    return GPT(config={
        "vocab_size": 32, "context_length": context_length, "embed_dim": 16,
        "num_heads": 4, "num_kv_heads": 2, "num_layers": 2, "ffn_dim": 32,
        "dropout": 0.0, "normalization": "rmsnorm", "position_encoding": "rope",
        "ffn_type": "swiglu", "fused_qkv": True,
    }).eval()


class Tokenizer:
    bos_id, eos_id, pad_id = 1, None, 0
    def encode(self, text): return [3 + ord(char) % 20 for char in text]
    def decode(self, ids): return "".join(chr(97 + value % 26) for value in ids)


def test_last_logits_equal_full_projection_last_position():
    model = tiny_model()
    ids = torch.tensor([[3, 4, 5, 6]])
    full = model(ids)
    last = model(ids, logits_mode="last")
    assert last.shape == (1, 1, model.vocab_size)
    torch.testing.assert_close(last, full[:, -1:])


def test_preallocated_cache_matches_full_and_supports_chunked_prefill():
    model = tiny_model()
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    full = model(ids)
    first, cache = model(ids[:, :3], use_cache=True, cache_capacity=8)
    second, cache = model(ids[:, 3:], kv_cache=cache, use_cache=True)
    assert all(isinstance(layer, PreallocatedKVCache) for layer in cache)
    assert all(layer.length == 5 and layer.position == 5 for layer in cache)
    torch.testing.assert_close(torch.cat((first, second), dim=1), full, rtol=1e-5, atol=1e-6)


def test_preallocated_cache_coexists_with_local_and_global_attention_layers():
    model = tiny_model()
    model.layers[0].attention.sliding_window = 3
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    full = model(ids)
    first, cache = model(ids[:, :3], use_cache=True, cache_capacity=8)
    second, cache = model(ids[:, 3:], kv_cache=cache, use_cache=True)
    assert not isinstance(cache[0], PreallocatedKVCache)
    assert isinstance(cache[1], PreallocatedKVCache)
    torch.testing.assert_close(torch.cat((first, second), dim=1), full, rtol=1e-5, atol=1e-6)


def test_generation_reserves_context_and_reports_dropped_prompt_tokens():
    result = generate_text(
        tiny_model(context_length=10), Tokenizer(), "abcdefghijklmnop",
        SamplingConfig(max_new_tokens=4, temperature=0, top_k=0, top_p=1,
                       repetition_penalty=1, no_repeat_ngram_size=0),
    )
    assert result.prompt_tokens == 16
    assert result.dropped_prompt_tokens == 10
    assert len(result.token_ids) == 4


def test_checkpoint_chat_contract_must_be_declared_or_explicit():
    with pytest.raises(ValueError, match="does not declare"):
        resolve_prompt_contract("chat", {})
    contract = resolve_prompt_contract(
        "chat", {"inference": {"prompt_contract": "aethyx-sft-v1"}}
    )
    assert contract.format_user_turn("Hello") == "<USER>\nHello\n<ASSISTANT>\n"
    assert resolve_inference_mode("auto", {}) == "base"
    assert resolve_inference_mode(
        "auto", {"inference": {"prompt_contract": "aethyx-sft-v1"}}
    ) == "chat"


def test_exact_sampling_does_not_block_copying():
    profile = sampling_for_decoding("exact", max_new_tokens=8)
    assert profile.temperature == 0
    assert profile.repetition_penalty == 1
    assert profile.no_repeat_ngram_size == 0


def test_sft_system_prefix_is_identified_for_context_preservation():
    prompt = "<SYSTEM>\nKeep invoice IDs exact.\n<USER>\nWhat is it?\n<ASSISTANT>\n"
    expected = len(Tokenizer().encode("<SYSTEM>\nKeep invoice IDs exact.\n"))
    assert critical_prefix_token_count(prompt, Tokenizer(), "aethyx-sft-v1") == expected
