"""
AethyxLM Tokenizer Configuration

Author: Aethyx Labs
"""

from pathlib import Path

# Root directory of tokenizer module
ROOT_DIR = Path(__file__).parent

# Dataset
DATA_DIR = ROOT_DIR / "data"
CORPUS_FILE = DATA_DIR / "corpus.txt"

# Output
TOKENIZER_FILE = ROOT_DIR / "tokenizer.json"
TOKENIZER_V2_FILE = ROOT_DIR / "tokenizer_v2.json"

# Tokenizer settings
VOCAB_SIZE = 32000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
]

STRUCTURAL_TOKENS = [
    "<DOC>",
    "<SYSTEM>",
    "<USER>",
    "<ASSISTANT>",
    "<TOOL>",
    "<CONTEXT>",
    "<MEMORY>",
]
