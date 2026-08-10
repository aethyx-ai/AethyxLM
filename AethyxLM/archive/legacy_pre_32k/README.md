# Legacy pre-32K archive

This directory contains the assets retired when AethyxLM switched to its 32K
multilingual tokenizer and fresh 31.17M-parameter training run on 2026-08-10.

- `checkpoints/`: all checkpoints trained with the 1,908-token vocabulary.
- `data/`: all pre-32K binary datasets, state files, and metadata. The raw
  `train.txt` and `val.txt` files were intentionally restored to active `data/`
  for fresh tokenization.
- `tokenizers/`: the 1,908-token tokenizer and the superseded 16K v2 experiment.

These assets are recoverable but incompatible with active 32K checkpoints and
dataset binaries. Do not move individual files back without restoring the
matching tokenizer, model configuration, and dataset sidecars together.
