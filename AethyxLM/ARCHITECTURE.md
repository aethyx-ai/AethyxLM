# AethyxLM Modern Transformer Architecture

> The executable model and JSON configuration are authoritative. Historical
> benchmark numbers later in this document predate the efficient-attention
> upgrade and should not be used as current performance claims.

## Current efficient baseline

The modern configuration now uses RMSNorm, RoPE, correctly budgeted SwiGLU,
grouped-query attention (8 query / 2 KV heads), fused QKV projection, PyTorch
scaled-dot-product attention, bias-free projections, and RoPE-aware KV-cached
decoding. Training supports gradient checkpointing, BF16-aware AMP, fused
AdamW, optional z-loss, and a staged context-length curriculum.

For longer-context experiments, layers can use a bounded sliding window with
periodic full-attention layers. Sliding layers retain only their configured
window in the KV cache while separately tracking absolute RoPE position. A
linear RoPE scaling factor is available for controlled continued-pretraining
experiments; accepting a longer tensor is not treated as proof of long-context
retrieval quality.

Legacy checkpoints remain compatible because models without the new config
keys retain separate Q/K/V projections and one KV head per query head.

## Compiled-context research boundary

Context compression remains experimental and disabled by default. The optional
`LatentContextAdapter` accepts typed features from any future local compiler
(text, graph, visual, structured, or hybrid), resamples a variable number of
items into a fixed latent budget, and exposes those latents to explicitly
selected decoder layers through gated cross-attention.

```text
compiler features + type IDs + validity mask
                    |
                    v
          fixed latent context budget
                    |
                    v
       selected decoder cross-attention layers
```

The adapter does not claim that a representation is information-preserving.
`evaluation/evaluator.py` reports reduction and downstream accuracy retention
as separate metrics. Compression succeeds only when task performance remains
within an agreed tolerance of the raw-context baseline.

`ContextMemoryBank` adds deterministic query-aware local retrieval before the
latent bottleneck. It preserves type IDs and source references, allowing a
future compiler to retrieve exact raw material when compressed context is not
sufficient. Retrieval, compression, and the decoder remain separate ablation
boundaries.

## Tokenizer generations

The checked-in tokenizer remains available for existing checkpoints. New
pretraining runs should create a versioned `tokenizer_v2.json`, which defaults
to NFKC normalization without lowercasing or accent stripping and reserves
document, role, tool, context, and memory tokens. A v2 tokenizer must be trained
on a representative multilingual corpus before it replaces the legacy model's
1,908-token vocabulary.

## Overview

This document describes the modern transformer architecture upgrades implemented in AethyxLM, moving from a GPT-2 style architecture to a modern LLM architecture.

## Changes Summary

### 1. Architecture Configuration
Extended `model/config.py` with new architecture options:
- `normalization`: `"layernorm"` | `"rmsnorm"` (default: `"layernorm"`)
- `position_encoding`: `"learned"` | `"rope"` (default: `"learned"`)
- `ffn_type`: `"gelu"` | `"swiglu"` (default: `"gelu"`)

Added new config constants:
- `ROPE_BASE = 10000.0` - Base frequency for RoPE
- `ROPE_MAX_SEQ_LEN = 8192` - Maximum sequence length for RoPE cache
- `SWIGLU_HIDDEN_MULT = 4` - Hidden dimension multiplier for SwiGLU

### 2. New Modules

#### `model/modules/rmsnorm.py`
- **RMSNorm**: Root Mean Square Layer Normalization
- More efficient than LayerNorm (no mean subtraction)
- Factory function `build_normalization()` for easy switching

#### `model/modules/rope.py`
- **RotaryEmbedding**: Rotary Positional Embeddings (RoPE)
- Precomputes and caches sin/cos values
- Automatic cache extension for longer sequences
- Applies RoPE to Q and K projections only
- Configurable base frequency and max sequence length

#### `model/modules/feedforward.py`
- **FeedForward**: Unified FFN module supporting both GELU and SwiGLU
- **SwiGLU**: `SiLU(x) * Gate(x)` with proper parameter scaling
- Factory function `build_feedforward()` for easy switching

### 3. Updated Components

#### `model/transformer_block.py`
- Uses `build_normalization()` for configurable normalization
- Uses new `FeedForward` module with configurable FFN type
- Compatible with both LayerNorm/RMSNorm and GELU/SwiGLU

#### `model/attention.py`
- Added RoPE integration
- Conditional RoPE application based on `position_encoding` config
- Maintains backward compatibility with learned positional embeddings

#### `model/transformer_block.py`
- Uses `build_normalization()` for configurable normalization
- Updated to use new `FeedForward` module

#### `model/gpt.py`
- Supports configurable architecture options
- Architecture summary printed at initialization
- Removed learned positional embeddings when using RoPE
- Added `_init_weights()` for proper weight initialization
- Added `from_checkpoint()` classmethod for checkpoint loading with architecture validation
- Architecture summary printed at initialization

#### `model/modules/rmsnorm.py`
- `RMSNorm` implementation with configurable epsilon
- `build_normalization()` factory function

#### `model/modules/rope.py`
- `RotaryEmbedding` class with automatic cache extension
- `apply_rotary_pos_emb()` helper function
- `build_rope()` factory function

#### `model/modules/feedforward.py`
- `FeedForward` class supporting both GELU and SwiGLU
- `build_feedforward()` factory function

#### `model/layers.py`
- Added weight initialization utilities
- Custom `Linear` layer with configurable initialization

#### `model/transformer_block.py`
- Updated to use configurable normalization and FFN type

#### `model/attention.py`
- Added RoPE integration
- Conditional RoPE application based on config

#### `model/gpt.py`
- Architecture summary printing
- Configurable positional embeddings (learned vs RoPE)
- Checkpoint loading with architecture validation
- Proper weight initialization

### 4. Configuration Files

#### `configs/train_config_modern.json`
Modern architecture configuration:
```json
{
  "model": {
    "normalization": "rmsnorm",
    "position_encoding": "rope",
    "ffn_type": "swiglu",
    ...
  }
}
```

#### `configs/datasets.json`
Dataset registry for mixed dataset training:
```json
{
  "tinystories": {"train": "data/train.bin", "val": "data/val.bin", "weight": 0.2},
  "fineweb_edu": {"train": "data/fineweb_train.bin", "val": "data/fineweb_val.bin", "weight": 0.8}
}
```

### 5. Training Pipeline Updates

#### `train.py`
- Supports mixed dataset loading via `MixedAethyxDataset`
- Auto-detects config format (legacy vs new)
- Automatic dataset preparation (TinyStories download, FineWeb-Edu check)
- DDP support with proper barriers

#### `dataset/dataset.py`
- Added `MixedAethyxDataset` for weighted multi-dataset sampling
- Weighted random sampling from multiple datasets

### 6. Data Preparation

#### `scripts/prepare_fineweb.py`
- Streaming FineWeb-Edu dataset download
- Text cleaning (NFKC normalization, whitespace normalization)
- SHA256-based deduplication
- Streaming tokenization (10M token chunks)
- Binary format output (uint16 memmap)
- Configurable target size (GB or document count)
- Metadata generation

### 3. Tests

Created comprehensive test suite:
- `tests/test_rmsnorm.py` - RMSNorm correctness
- `tests/test_rope.py` - RoPE forward, cache, correctness
- `tests/test_feedforward.py` - GELU/SwiGLU forward, gradients, parameters

### 2. Benchmarking

Created `benchmark_architecture.py` for comparing:
- GPT-2 style (LayerNorm + Learned PosEmb + GELU)
- Modern (RMSNorm + RoPE + SwiGLU)

Metrics measured:
- Forward pass latency
- Training step latency
- Peak memory usage
- Training throughput (tokens/sec)

### 3. Configuration Files

- `configs/train_config_kaggle.json` - Legacy GPT-2 style config
- `configs/train_config_modern.json` - Modern architecture config
- `configs/datasets.json` - Dataset registry with weights

### 4. Testing

All tests pass:
- RMSNorm: forward pass, normalization correctness, LayerNorm compatibility
- RoPE: forward pass, output shapes, cache extension, apply_rotary_pos_emb, cache correctness
- FeedForward: GELU/SwiGLU forward, mathematical correctness, gradient flow, parameter counts

### 5. Benchmark Results (CPU)

| Metric | GPT-2 Style | Modern | Speedup |
|--------|-------------|--------|---------|
| Parameters | 1.0M | 8.9M | 8.5x more |
| Forward Time | 41ms | 174ms | 0.24x |
| Train Step | 140ms | 606ms | 0.23x |
| Throughput | 3,670 tok/s | 845 tok/s | 0.23x |

Note: Modern architecture is slower on CPU due to larger model size (8.5M vs 1.0M params) and additional computations (RoPE, SwiGLU). On GPU with proper parallelization, modern architecture typically shows better scaling.

### 5. Usage

#### Training with Modern Architecture
```bash
# Prepare FineWeb-Edu (10GB)
python scripts/prepare_fineweb.py --target-gb 10

# Resume training from checkpoint
python train.py --config configs/train_config_modern.json --device cuda

# Or start fresh
python train.py --config configs/train_config_modern.json --device cuda
```

#### Training with Legacy Architecture
```bash
python train.py --config configs/train_config_kaggle.json --device cuda
```

## Mathematical Background

### RMSNorm
```
RMS(x) = sqrt(mean(x^2) + eps)
y = x / RMS(x) * weight
```

### RoPE
For position `t` and dimension `i`:
```
theta_i = base^(-2i/d)
R_t = rotation_matrix(t * theta_i)
q_rot = q * cos(t*theta) + rotate_half(q) * sin(t*theta)
```

### SwiGLU
```
SwiGLU(x) = SiLU(xW_gate) * (xW_value)
where SiLU(x) = x * sigmoid(x)
```

## References

## Empirical pilot status (2026-08-09)

These results are bounded engineering pilots, not frontier-model claims:

- CUDA environment: PyTorch `2.11.0+cu130` on an NVIDIA GeForce MX450 (SM 7.5).
  Memory-efficient SDPA and math execute successfully. This Windows wheel has no
  FlashAttention kernel, and cuDNN attention does not support SM 7.5.
- GPU execution benchmark: the modern path reached 3,770 training tokens/s and
  1.10x cached-decoding speedup in the small 128-token test. Kernel-launch
  overhead dominates at this scale; see `gpu_benchmark.json`.
- Tokenizer v2: trained a 16K vocabulary on a bounded 24-variety English/Indic
  sample. On the in-sample diagnostic it produced 274,362 tokens versus 545,599
  for the legacy tokenizer, with zero unknown tokens and 100% normalized
  round-trip. This is not a held-out language-quality estimate.
- Scaling feasibility: 10.17M and 23.53M configurations completed equal 15,360
  token CUDA pilots. The 76.98M and 272.54M configurations exceed the safe AdamW
  capacity of the 2 GB GPU and were capacity-gated.
- Matched-token architecture pilot: after 51,200 tokens, the 8.09M modern model
  reached validation loss 4.894 versus 5.041 for the 6.84M classic model, while
  running slower on this GPU. The differing parameter counts and short budget
  prevent an intelligence conclusion.
- Context compression: a synthetic exact-binding task rejected the current
  generic latent resampler. At 70.3% unit reduction it achieved 7.83% exact
  retrieval (random baseline 6.25%). Therefore the proposed 70% context reduction
  remains unvalidated and requires key/address-preserving compiler structure.

Reproducible artifacts are `sdpa_backend_probe.json`, `gpu_benchmark.json`,
`tokenizer/tokenizer_v2_evaluation.json`, `scaling_pilot_gpu.json`,
`architecture_quality_pilot_gpu.json`, and `context_compression_pilot.json`.

- [RMSNorm](https://arxiv.org/abs/1910.07467) - Zhang & Sennrich, 2019
- [RoPE](https://arxiv.org/abs/2104.09864) - Su et al., 2021
- [SwiGLU](https://arxiv.org/abs/2002.05202) - Shazeer, 2020
- [Llama 3 Architecture](https://ai.meta.com/blog/meta-llama-3/) - Meta AI, 2024
