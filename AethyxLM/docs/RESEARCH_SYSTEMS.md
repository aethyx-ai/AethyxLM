# AethyxLM Research Systems

This document is the map for capabilities that surround the stable pretraining pipeline. Each system owns its own code, configuration, outputs, and tests so experiments do not become mixed with model architecture work.

## Stable pretraining

- Entry point: `train.py`
- Configuration: `configs/train_config_modern.json`
- Rotating checkpoints: `checkpoints/`
- Non-rotating milestones: `checkpoints/milestones/`
- Machine-readable metrics: `logs/metrics.jsonl`

The model architecture remains unchanged. The trainer additions are optional logging and milestone hooks only.

## Checkpoint milestones

Automatic milestones are preserved every 10,000 steps by the modern configuration. They are outside numbered-checkpoint rotation.

Archive an existing checkpoint manually:

```powershell
python scripts/manage_milestones.py archive checkpoints/checkpoint_latest.pt
python scripts/manage_milestones.py list
```

## Evaluation

`scripts/evaluate_checkpoint_suite.py` measures:

- validation loss and perplexity for every configured source;
- weighted aggregate loss;
- deterministic multilingual generation and repetition diagnostics;
- synthetic long-context passkey retrieval at multiple depths;
- unique checkpoint steps, avoiding duplicate `latest`/numbered aliases.

Example:

```powershell
python scripts/evaluate_checkpoint_suite.py `
  --checkpoint-dir checkpoints `
  --output evaluation/results/local-checkpoints.json
```

Generation diagnostics are not presented as a complete coherence score. Human evaluation and established downstream benchmarks remain necessary.

## Experiment tracking

Training writes append-only JSONL events locally. No hosted tracking account is required and no training data is logged.

```powershell
python scripts/summarize_experiments.py logs/metrics.jsonl `
  --output evaluation/results/training-summary.json
```

## Instruction fine-tuning

SFT is deliberately separate from pretraining:

- Data preparation: `scripts/prepare_sft_data.py`
- Dataset and assistant-only masking: `training/sft_dataset.py`
- Configuration: `configs/sft_config.json`
- Entry point: `sft_train.py`
- Checkpoints: `checkpoints/sft/`
- Logs: `logs/sft/`

Accepted input forms are OpenAI-style `messages`, ShareGPT-style `conversations`, and `prompt`/`response` pairs. Preparation normalizes roles, filters malformed or highly repetitive examples, deduplicates content, and produces deterministic train/validation splits.

```powershell
python scripts/prepare_sft_data.py instructions.jsonl --output-dir data/sft
python sft_train.py --config configs/sft_config.json `
  --base-checkpoint checkpoints/milestones/checkpoint_step_100000.pt
```

Only assistant content contributes to the loss. System and user tokens remain context but are masked with `-100` targets.

## Inference

Reusable generation lives in `inference/generation.py`. It provides:

- KV-cached decoding;
- greedy or temperature sampling;
- top-k, top-p, and min-p filtering;
- repetition penalties;
- stop strings and streaming callbacks.

`chat.py` uses this engine and retains checkpoint selection and tokenizer-fingerprint verification.

## Context representation research

The isolated research package is `context_lab/`. It contains graph extraction, verbatim-risk guards, visual-page planning, benchmark metrics, and a browser Web Worker reference implementation. See `CONTEXT_COMPRESSION_RESEARCH.md` for the design and limitations.

## Intentionally excluded

Model export and deployment infrastructure are not part of this phase.

