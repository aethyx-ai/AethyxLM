"""Short, bounded inference benchmark for optional quantization modes."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import torch

from inference.quantization import quantization_coverage, quantize_model_for_inference
from model.gpt import GPT


def serialized_bytes(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def timed_forward(model, tokens, iterations):
    with torch.inference_mode():
        for _ in range(2):
            model(tokens, logits_mode="last")
        start = time.perf_counter()
        for _ in range(iterations):
            model(tokens, logits_mode="last")
    return (time.perf_counter() - start) / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 100:
        raise ValueError("iterations must be between 1 and 100")
    if args.checkpoint:
        baseline = GPT.from_checkpoint(str(args.checkpoint), device="cpu").eval()
    else:
        baseline = GPT(config={
            "vocab_size": 512, "context_length": 128, "embed_dim": 64,
            "num_heads": 4, "num_kv_heads": 2, "num_layers": 2,
            "ffn_dim": 128, "dropout": 0.0, "normalization": "rmsnorm",
            "position_encoding": "rope", "ffn_type": "swiglu", "fused_qkv": True,
        }).eval()
    length = min(args.sequence_length, baseline.context_length)
    tokens = torch.randint(0, baseline.vocab_size, (1, length))
    with torch.inference_mode():
        reference = baseline(tokens, logits_mode="last").float()
    baseline_seconds = timed_forward(baseline, tokens, args.iterations)
    baseline_size = serialized_bytes(baseline)
    coverage_before = quantization_coverage(baseline)

    candidates = {}
    for mode in ("dynamic-int8-ffn", "dynamic-int8"):
        quantized = quantize_model_for_inference(baseline, mode, "cpu").eval()
        with torch.inference_mode():
            candidate = quantized(tokens, logits_mode="last").float()
        quantized_seconds = timed_forward(quantized, tokens, args.iterations)
        candidates[mode] = {
            "latency_ms": quantized_seconds * 1000,
            "speedup": baseline_seconds / quantized_seconds,
            "serialized_bytes": serialized_bytes(quantized),
            "coverage": quantization_coverage(quantized),
            "max_abs_logit_error": float((reference - candidate).abs().max()),
            "mean_abs_logit_error": float((reference - candidate).abs().mean()),
            "cosine_similarity": float(torch.nn.functional.cosine_similarity(reference, candidate, dim=-1).mean()),
            "top1_agreement": float((reference.argmax(-1) == candidate.argmax(-1)).float().mean()),
        }
    payload = {
        "device": "cpu",
        "sequence_length": length,
        "iterations": args.iterations,
        "baseline_ms": baseline_seconds * 1000,
        "baseline_serialized_bytes": baseline_size,
        "coverage_before": coverage_before,
        "candidates": candidates,
        "warning": "A short CPU microbenchmark is not a task-quality evaluation or a GPU result.",
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
