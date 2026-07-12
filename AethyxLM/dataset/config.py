"""
AethyxLM Dataset Configuration
"""

from pathlib import Path

ROOT_DIR = Path(__file__).parent

# Path to training corpus
DATA_FILE = ROOT_DIR / "corpus.txt"

# Context window (sequence length)
CONTEXT_LENGTH = 128

# Batch size
BATCH_SIZE = 8

# Number of workers
NUM_WORKERS = 0

# Shuffle training data
SHUFFLE = True