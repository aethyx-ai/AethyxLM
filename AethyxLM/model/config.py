"""
Model configuration for AethyxLM.
"""

# ==========================
# Tokenizer
# ==========================

VOCAB_SIZE = 32000

# ==========================
# Sequence
# ==========================

CONTEXT_LENGTH = 128

# ==========================
# Model Architecture
# ==========================

EMBED_DIM = 256
NUM_HEADS = 8

assert EMBED_DIM % NUM_HEADS == 0, (
    "EMBED_DIM must be divisible by NUM_HEADS."
)

HEAD_DIM = EMBED_DIM // NUM_HEADS

NUM_LAYERS = 8
DROPOUT = 0.1

# ==========================
# Feed Forward Network
# ==========================

FFN_DIM = EMBED_DIM * 4

USE_BIAS = True

# LayerNorm

LAYER_NORM_EPS = 1e-5