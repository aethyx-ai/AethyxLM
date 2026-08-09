"""Benchmark classic and modern AethyxLM execution paths."""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer


ROOT = Path(__file__).resolve().parent


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(action, steps, device):
    synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        action()
    synchronize(device)
    return (time.perf_counter() - start) / steps


def benchmark_model(
    config, name, device, steps=20, batch_size=4, seq_len=64, decode_tokens=16
):
    config = config.copy()
    config["gradient_checkpointing"] = False
    model = GPT(config=config).to(device)
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def autocast():
        return (
            torch.autocast("cuda", dtype=amp_dtype)
            if use_amp
            else nullcontext()
        )

    def forward(input_ids):
        with autocast():
            return model(input_ids)
    tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, model.vocab_size, tokens.shape, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(3):
            forward(tokens)
        forward_seconds = timed(lambda: forward(tokens), steps, device)

        prefix = tokens[:1]

        def uncached_decode():
            sequence = prefix
            for _ in range(decode_tokens):
                logits = forward(sequence[:, -model.context_length :])
                sequence = torch.cat((sequence, logits[:, -1].argmax(-1, keepdim=True)), 1)

        def cached_decode():
            with autocast():
                logits, cache = model(prefix, use_cache=True)
            for _ in range(decode_tokens):
                next_token = logits[:, -1].argmax(-1, keepdim=True)
                with autocast():
                    logits, cache = model(next_token, kv_cache=cache, use_cache=True)

        uncached_seconds = timed(uncached_decode, max(2, steps // 4), device)
        cached_seconds = timed(cached_decode, max(2, steps // 4), device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(tokens)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(2):
        train_step()
    train_seconds = timed(train_step, steps, device)
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    return {
        "name": name,
        "compute_dtype": "float16" if use_amp else "float32",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "forward_ms": forward_seconds * 1000,
        "train_step_ms": train_seconds * 1000,
        "training_tokens_per_second": batch_size * seq_len / train_seconds,
        "decode_tokens": decode_tokens,
        "uncached_decode_ms": uncached_seconds * 1000,
        "cached_decode_ms": cached_seconds * 1000,
        "kv_cache_decode_speedup": uncached_seconds / cached_seconds,
        "peak_memory_mb": peak_memory / 1024**2,
    }


def load_model_config(path):
    return json.loads(path.read_text(encoding="utf-8"))["model"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AethyxTokenizer(ROOT / "tokenizer" / "tokenizer.json")

    configs = {
        "classic": load_model_config(ROOT / "configs" / "train_config_kaggle.json"),
        "modern": load_model_config(ROOT / "configs" / "train_config_modern.json"),
    }
    for config in configs.values():
        config["vocab_size"] = tokenizer.vocab_size
        config["context_length"] = max(
            config["context_length"], args.seq_len + args.decode_tokens
        )

    results = {
        name: benchmark_model(
            config,
            name,
            device,
            args.steps,
            args.batch_size,
            args.seq_len,
            args.decode_tokens,
        )
        for name, config in configs.items()
    }
    payload = {"device": str(device), "results": results}
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
