"""
AethyxLM Training Configuration
"""

# Model
MODEL_CONFIG = {
    "vocab_size": 32000,
    "context_length": 128,
    "embed_dim": 256,
    "num_heads": 8,
    "num_layers": 8,
    "ffn_dim": 1024,
    "dropout": 0.1,
    "use_bias": True,
    "layer_norm_eps": 1e-5,
}

# Training
TRAIN_CONFIG = {
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "grad_clip": 1.0,
    "warmup_steps": 1000,
    "max_steps": 10000,
    "min_lr_ratio": 0.1,
    "grad_accum_steps": 1,
    "use_amp": True,
    "batch_size": 8,
    "num_workers": 0,
}

# Data
DATA_CONFIG = {
    "train_file": "data/train.txt",
    "val_file": "data/val.txt",
    "context_length": 128,
}

# Checkpointing
CHECKPOINT_CONFIG = {
    "checkpoint_dir": "checkpoints",
    "log_interval": 10,
    "eval_interval": 500,
    "save_interval": 1000,
}

# Tokenizer
TOKENIZER_CONFIG = {
    "tokenizer_file": "tokenizer/tokenizer.json",
    "vocab_size": 32000,
}