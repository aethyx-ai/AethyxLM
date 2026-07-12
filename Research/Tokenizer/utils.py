"""
AethyxLM

Module:
utils.py

Purpose:
Utility functions for tokenizer.

Author:
Aethyx Labs
"""

import re


def preprocess(text: str) -> str:
    """
    Normalize text before tokenization.
    """

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def split_word(word: str):
    """
    Split a word into characters.
    """

    return list(word)