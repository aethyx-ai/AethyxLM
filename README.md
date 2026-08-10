# AethyxLM

> Building efficient frontier AI systems from India.

AethyxLM is a decoder-only language model and training stack built from scratch in PyTorch by Aethyx AI. The project currently focuses on a stable multilingual pretraining pipeline while separately researching more information-dense ways for transformers to consume context.

## Current production profile

- 31,171,968 parameters
- 32,000-token multilingual ByteLevel BPE tokenizer
- 12 transformer layers, 384-dimensional embeddings
- 6 query heads and 2 key/value heads using grouped-query attention (GQA)
- RoPE, RMSNorm, SwiGLU, fused QKV projections and QK normalization
- PyTorch scaled-dot-product attention (SDPA)
- Gradient checkpointing, mixed precision and fused AdamW when supported
- Sequence-length curriculum: 128, then 256, then 512 tokens
- KV-cached generation and optional sliding-window/global attention experiments

The active configuration is [`AethyxLM/configs/train_config_modern.json`](AethyxLM/configs/train_config_modern.json).

## Training data

The active storage-bounded mixture contains approximately 526 million tokenized tokens across 15 datasets:

- TinyStories
- FineWeb-Edu
- OpenWebMath
- Cosmopedia OpenStax
- Cosmopedia Khan Academy
- IndicCorp v2 data for Hindi, Bengali, Telugu, Tamil, Marathi, Gujarati, Kannada, Malayalam, Punjabi and Urdu

Remote datasets are streamed and tokenized directly into capped binary files, avoiding full dataset downloads. Each dataset has independent preparation state, validation data and tokenizer-fingerprint metadata. Source revisions are pinned for reproducibility.

Legacy pre-32K tokenizer artifacts and checkpoints are isolated under [`AethyxLM/archive/legacy_pre_32k`](AethyxLM/archive/legacy_pre_32k).

## Quick start

Install dependencies:

```powershell
cd AethyxLM
python -m pip install -r requirements.txt
```

Verify that the tokenizer, binary datasets, metadata and model configuration agree:

```powershell
python scripts/check_training_readiness.py
```

Start production training on CUDA:

```powershell
python train.py --config configs/train_config_modern.json --device cuda
```

The default run performs 100,000 optimizer steps. On a 6 GB RTX 3050 Laptop GPU, the initial estimate is roughly 10-20 hours, but actual throughput depends on GPU power limits and thermal throttling.

## Checkpoints and resume

Training saves a complete resumable checkpoint every 1,000 optimizer steps. A checkpoint includes the model, optimizer, learning-rate scheduler, AMP scaler, training progress, token count and random-number-generator states.

Resume from the latest checkpoint with:

```powershell
python train.py --config configs/train_config_modern.json --device cuda --resume checkpoints/checkpoint_latest.pt
```

The checkpoint directory retains:

- `checkpoint_latest.pt`
- `checkpoint_best.pt` after validation improves
- The three most recent numbered step checkpoints

An interruption between save boundaries can lose up to 999 optimizer steps.

## Preparing the storage-bounded dataset bundle

The prepared binaries can be copied to another training machine; the tokenizer does not need to be retrained. To recreate or extend the bundle when needed:

```powershell
python scripts/prepare_dataset_bundle.py --bundle multilingual_v2_32k
```

For training on another device, copy the project code, `AethyxLM/tokenizer`, `AethyxLM/configs`, and the active `.bin` plus `.bin.meta.json` files in `AethyxLM/data`. Copy `AethyxLM/checkpoints` as well when resuming an existing run.

## Context-efficiency research

AethyxLM's intended long-term differentiation is a context compilation layer between user information and the transformer:

```text
Text, tools, documents and agent state
                |
       Local context compiler
                |
 Compact machine-readable representation
                |
             AethyxLM
```

The research question is not whether images are inherently better than tokens. It is which representation—text, graphs, spatial or visual layouts, learned latents, or a hybrid—preserves the same useful information with less bandwidth, memory and inference compute.

The codebase includes an opt-in, representation-agnostic context adapter and benchmarking infrastructure. Context compression is not yet a validated product capability, and the proposed reduction of up to 70% remains a hypothesis that must be tested against task accuracy, retrieval, tool use, latency, memory, bandwidth and cost.

This research remains isolated from the stable language-model training pipeline.

## Project structure

```text
AethyxLM/
|-- model/          Transformer architecture
|-- training/       Trainer, optimizer and scheduling logic
|-- tokenizer/      Active 32K tokenizer and metadata
|-- data/           Tokenized training/validation binaries
|-- dataset/        Dataset loading and mixing
|-- evaluation/     Model and compression evaluation
|-- configs/        Production model and dataset configurations
|-- scripts/        Dataset preparation and readiness tools
|-- checkpoints/    Active resumable training checkpoints
|-- archive/        Legacy artifacts kept out of the active path
|-- train.py        Training entry point
`-- chat.py         Local inference interface
```

More architectural detail is available in [`AethyxLM/ARCHITECTURE.md`](AethyxLM/ARCHITECTURE.md).

## Status and roadmap

Completed foundations include the custom tokenizer, modern decoder architecture, mixed-dataset pipeline, storage-capped streaming preparation, checkpoint/resume support and CUDA smoke testing.

Current priorities are:

1. Train and evaluate the 31M multilingual baseline.
2. Run controlled scaling experiments.
3. Improve long-context efficiency and retrieval.
4. Benchmark graph, visual, latent and hybrid context representations.
5. Scale only when experiments justify the added compute.

## Contributing

Research discussions and contributions around efficient transformers, multilingual training, evaluation and context representation are welcome through GitHub issues and discussions.

## License

This project is licensed under the MIT License.
