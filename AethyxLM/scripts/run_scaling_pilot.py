"""Run short, controlled CUDA scaling pilots on the existing token corpus.

This is a systems/learning-curve experiment, not evidence of final model
intelligence or a substitute for compute-optimal pretraining runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.gpt import GPT


MODEL_SPECS = {
    "10m": {"embed_dim": 384, "num_heads": 6, "num_kv_heads": 2, "num_layers": 6, "ffn_dim": 1024},
    "30m": {"embed_dim": 512, "num_heads": 8, "num_kv_heads": 2, "num_layers": 8, "ffn_dim": 1408},
    "100m": {"embed_dim": 768, "num_heads": 12, "num_kv_heads": 4, "num_layers": 12, "ffn_dim": 2048},
    "300m": {"embed_dim": 1024, "num_heads": 16, "num_kv_heads": 4, "num_layers": 24, "ffn_dim": 2816},
    "classic_7m": {
        "embed_dim": 256, "num_heads": 8, "num_kv_heads": 8,
        "num_layers": 8, "ffn_dim": 1024, "normalization": "layernorm",
        "position_encoding": "learned", "ffn_type": "gelu", "use_bias": True,
        "fused_qkv": False, "qk_norm": False,
    },
    "modern_7m": {
        "embed_dim": 256, "num_heads": 8, "num_kv_heads": 2,
        "num_layers": 8, "ffn_dim": 1024,
    },
}


def model_config(spec: dict, vocab_size: int, context_length: int) -> dict:
    return {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "dropout": 0.0,
        "normalization": "rmsnorm",
        "position_encoding": "rope",
        "ffn_type": "swiglu",
        "use_bias": False,
        "fused_qkv": True,
        "use_sdpa": True,
        "qk_norm": True,
        "gradient_checkpointing": False,
        **spec,
    }


def batches(data: np.memmap, indices: np.ndarray, batch_size: int, length: int, device: str):
    offsets = np.arange(length + 1)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        array = np.asarray(data[selected[:, None] + offsets[None, :]], dtype=np.int64)
        tensor = torch.from_numpy(array).to(device, non_blocking=True)
        yield tensor[:, :-1], tensor[:, 1:]


@torch.no_grad()
def validation_loss(model, data, indices, batch_size, length, amp_dtype) -> float:
    model.eval()
    total = 0.0
    tokens = 0
    for inputs, targets in batches(data, indices, batch_size, length, "cuda"):
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
            )
        total += float(loss)
        tokens += targets.numel()
    model.train()
    return total / tokens


def run_one(name: str, spec: dict, args, train_data, val_data, rng) -> dict:
    config = model_config(spec, args.vocab_size, args.context_length)
    # Construct on CPU first so the exact parameter count can be capacity-gated.
    model = GPT(config=config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    # AdamW training commonly needs roughly 16 bytes/parameter across weights,
    # gradients, and optimizer states. Preserve headroom for CUDA and activations.
    estimated_bytes = parameter_count * 16 + 256 * 1024**2
    if estimated_bytes > free_bytes * args.max_vram_fraction:
        del model
        return {
            "status": "skipped_capacity_gate",
            "parameters": parameter_count,
            "estimated_training_bytes": estimated_bytes,
            "free_device_bytes": free_bytes,
        }

    model.cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), fused=True
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    train_indices = rng.integers(
        0, len(train_data) - args.context_length - 1,
        size=args.steps * args.batch_size * args.grad_accumulation,
    )
    val_indices = np.linspace(
        0,
        len(val_data) - args.context_length - 2,
        args.eval_batches * args.batch_size,
        dtype=np.int64,
    )
    initial_loss = validation_loss(
        model, val_data, val_indices, args.batch_size, args.context_length, amp_dtype
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    cursor = 0
    try:
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for _ in range(args.grad_accumulation):
                selected = train_indices[cursor : cursor + args.batch_size]
                cursor += args.batch_size
                inputs, targets = next(
                    batches(train_data, selected, args.batch_size, args.context_length, "cuda")
                )
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                    ) / args.grad_accumulation
                scaler.scale(loss).backward()
                accumulated += float(loss.detach()) * args.grad_accumulation
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(accumulated / args.grad_accumulation)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        final_loss = validation_loss(
            model, val_data, val_indices, args.batch_size, args.context_length, amp_dtype
        )
        trained_tokens = args.steps * args.batch_size * args.grad_accumulation * args.context_length
        return {
            "status": "completed",
            "parameters": parameter_count,
            "initial_validation_loss": initial_loss,
            "final_validation_loss": final_loss,
            "validation_perplexity": math.exp(min(final_loss, 20)),
            "mean_training_loss": sum(losses) / len(losses),
            "last_training_loss": losses[-1],
            "optimizer_steps": args.steps,
            "trained_tokens": trained_tokens,
            "tokens_per_second": trained_tokens / elapsed,
            "elapsed_seconds": elapsed,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "amp_dtype": str(amp_dtype),
        }
    except torch.OutOfMemoryError as error:
        return {"status": "cuda_out_of_memory", "parameters": parameter_count, "error": str(error)}
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.validation_data, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(args.seed)
    result = {
        "experiment": "short controlled scaling pilot",
        "warning": "Short equal-token runs validate feasibility and learning behavior, not frontier intelligence or a scaling law.",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "settings": vars(args),
        "runs": {},
    }
    for name in args.models.split(","):
        name = name.strip().lower()
        if name not in MODEL_SPECS:
            raise ValueError(f"Unknown model size {name}")
        print(f"Running {name} pilot", flush=True)
        result["runs"][name] = run_one(
            name, MODEL_SPECS[name], args, train_data, val_data, rng
        )
        print(json.dumps({name: result["runs"][name]}, indent=2), flush=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/train.bin")
    parser.add_argument("--validation-data", default="data/val.bin")
    parser.add_argument("--vocab-size", type=int, default=1908)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-vram-fraction", type=float, default=0.82)
    parser.add_argument("--models", default="10m,30m,100m,300m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="scaling_pilot_gpu.json")
    main(parser.parse_args())
