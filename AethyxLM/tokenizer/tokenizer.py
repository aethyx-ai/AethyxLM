"""
AethyxLM
Production Tokenizer

Author: Aethyx Labs
"""

from pathlib import Path
import hashlib

from tokenizers import Tokenizer

from tokenizer.config import TOKENIZER_FILE


class AethyxTokenizer:
    """
    Wrapper around the Hugging Face BPE tokenizer.
    """

    def __init__(self, tokenizer_path=TOKENIZER_FILE):

        tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found:\n{tokenizer_path}\n\n"
                "Run train_tokenizer.py first."
            )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.path = tokenizer_path.resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def encode(self, text: str):
        """
        Convert text into token IDs.
        """

        encoding = self.tokenizer.encode(text)

        return encoding.ids

    def decode(self, ids):
        """
        Convert token IDs back into text.
        """

        return self.tokenizer.decode(ids)

    def tokenize(self, text: str):
        """
        Return token strings instead of IDs.
        """

        encoding = self.tokenizer.encode(text)

        return encoding.tokens

    def token_to_id(self, token: str):
        """Return the integer ID for a token, or ``None`` when absent."""
        return self.tokenizer.token_to_id(token)

    @property
    def eos_id(self):
        return self.token_to_id("<EOS>")

    @property
    def bos_id(self):
        return self.token_to_id("<BOS>")

    @property
    def pad_id(self):
        return self.token_to_id("<PAD>")

    @property
    def vocab_size(self):
        """
        Vocabulary size.
        """

        return self.tokenizer.get_vocab_size()
