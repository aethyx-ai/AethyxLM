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
NUM_KV_HEADS = NUM_HEADS

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
FUSED_QKV = False
USE_SDPA = True
QK_NORM = False

# LayerNorm
LAYER_NORM_EPS = 1e-5

# ==========================
# Architecture Options
# ==========================

# Normalization: "layernorm" | "rmsnorm"
NORMALIZATION = "layernorm"

# Position Encoding: "learned" | "rope"
POSITION_ENCODING = "learned"

# Feed Forward Network Type: "gelu" | "swiglu"
FFN_TYPE = "gelu"

# RoPE Configuration
ROPE_BASE = 10000.0
ROPE_MAX_SEQ_LEN = 8192

# SwiGLU Configuration
SWIGLU_HIDDEN_MULT = 4  # hidden_dim = embed_dim * mult * 2/3 for SwiGLU
