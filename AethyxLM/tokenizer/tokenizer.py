"""
AethyxLM
Production Tokenizer

Author: Aethyx Labs
"""

from pathlib import Path

from tokenizers import Tokenizer

from .config import TOKENIZER_FILE


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

    @property
    def vocab_size(self):
        """
        Vocabulary size.
        """

        return self.tokenizer.get_vocab_size()