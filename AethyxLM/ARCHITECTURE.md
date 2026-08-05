# AethyxLM Modern Transformer Architecture

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

- [RMSNorm](https://arxiv.org/abs/1910.07467) - Zhang & Sennrich, 2019
- [RoPE](https://arxiv.org/abs/2104.09864) - Su et al., 2021
- [SwiGLU](https://arxiv.org/abs/2002.05202) - Shazeer, 2020
- [Llama 3 Architecture](https://ai.meta.com/blog/meta-llama-3/) - Meta AI, 2024