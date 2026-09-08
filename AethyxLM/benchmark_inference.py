"""Bounded benchmark for final-position logits and KV-cache append strategies."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from model.gpt import GPT


def timed(action, iterations):
    with torch.inference_mode():
        action()
        start = time.perf_counter()
        for _ in range(iterations):
            action()
    return (time.perf_counter() - start) / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 100 or not 1 <= args.decode_tokens <= 128:
        raise ValueError("iterations must be 1-100 and decode-tokens must be 1-128")
    model = (
        GPT.from_checkpoint(str(args.checkpoint), device="cpu").eval()
        if args.checkpoint
        else GPT(config={
            "vocab_size": 512, "context_length": 128, "embed_dim": 64,
            "num_heads": 4, "num_kv_heads": 2, "num_layers": 2,
            "ffn_dim": 128, "dropout": 0.0, "normalization": "rmsnorm",
            "position_encoding": "rope", "ffn_type": "swiglu", "fused_qkv": True,
        }).eval()
    )
    length = min(args.sequence_length, model.context_length - args.decode_tokens)
    if length <= 0:
        raise ValueError("sequence plus decode budget exceeds context_length")
    prompt = torch.randint(0, model.vocab_size, (1, length))
    continuation = torch.randint(0, model.vocab_size, (1, args.decode_tokens))

    full_seconds = timed(lambda: model(prompt), args.iterations)
    last_seconds = timed(lambda: model(prompt, logits_mode="last"), args.iterations)

    def decode(preallocated):
        capacity = model.context_length if preallocated else None
        _, cache = model(
            prompt, use_cache=True, logits_mode="last", cache_capacity=capacity
        )
        for index in range(args.decode_tokens):
            _, cache = model(
                continuation[:, index : index + 1],
                kv_cache=cache,
                use_cache=True,
                logits_mode="last",
            )

    tuple_seconds = timed(lambda: decode(False), args.iterations)
    preallocated_seconds = timed(lambda: decode(True), args.iterations)
    payload = {
        "device": "cpu",
        "sequence_length": length,
        "decode_tokens": args.decode_tokens,
        "iterations": args.iterations,
        "full_logits_ms": full_seconds * 1000,
        "last_logits_ms": last_seconds * 1000,
        "last_logits_speedup": full_seconds / last_seconds,
        "tuple_cache_decode_ms": tuple_seconds * 1000,
        "preallocated_cache_decode_ms": preallocated_seconds * 1000,
        "preallocated_cache_speedup": tuple_seconds / preallocated_seconds,
        "warning": "Short CPU timings vary by host; benchmark the deployment GPU before claiming a gain.",
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
