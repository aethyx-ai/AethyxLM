"""
Model configuration for AethyxLM.
"""

# Tokenizer
VOCAB_SIZE = 32000

# Sequence
CONTEXT_LENGTH = 128

# Model Architecture
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 8
DROPOUT = 0.1

# Feed Forward Network
FFN_DIM = EMBED_DIM * 4