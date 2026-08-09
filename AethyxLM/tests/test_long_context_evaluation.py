import torch

from evaluation.long_context import build_passkey_example, evaluate_passkey_retrieval
from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer


def test_passkey_examples_have_controlled_context_length():
    tokenizer = AethyxTokenizer()
    prompt, answer = build_passkey_example(tokenizer, 64, 0.5, 123456)
    assert len(prompt) == 64
    assert answer


def test_long_context_benchmark_returns_every_length_depth_cell():
    tokenizer = AethyxTokenizer()
    model = GPT(
        config={
            "vocab_size": tokenizer.vocab_size,
            "context_length": 64,
            "embed_dim": 16,
            "num_heads": 4,
            "num_layers": 1,
            "ffn_dim": 32,
            "dropout": 0.0,
            "position_encoding": "rope",
        }
    )
    results = evaluate_passkey_retrieval(
        model, tokenizer, context_lengths=[32, 48], depths=[0.25, 0.75], trials=1
    )
    assert len(results) == 4
    assert all(result.trials == 1 for result in results)
    assert all(0.0 <= result.accuracy <= 1.0 for result in results)
