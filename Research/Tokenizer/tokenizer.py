"""
AethyxLM

Module:
tokenizer.py

Purpose:
AethyxTokenizer Version 0.2

Author:
Aethyx Labs

Version:
v0.2 Spark
"""

import json
import re


class AethyxTokenizer:

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(self):

        self.word_to_id = {}
        self.id_to_word = {}

    def preprocess(self, text):

        text = text.lower()

        text = re.sub(r"([.,!?;:()])", r" \1 ", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def build_vocab(self, text):

        text = self.preprocess(text)

        words = text.split()

        unique_words = sorted(set(words))

        vocabulary = [
            self.PAD,
            self.UNK,
            self.BOS,
            self.EOS
        ] + unique_words

        self.word_to_id = {
            word: idx
            for idx, word in enumerate(vocabulary)
        }

        self.id_to_word = {
            idx: word
            for word, idx in self.word_to_id.items()
        }

    def encode(self, text):

        text = self.preprocess(text)

        words = text.split()

        ids = [self.word_to_id[self.BOS]]

        for word in words:

            ids.append(
                self.word_to_id.get(
                    word,
                    self.word_to_id[self.UNK]
                )
            )

        ids.append(self.word_to_id[self.EOS])

        return ids

    def decode(self, ids):

        words = []

        for idx in ids:

            word = self.id_to_word[idx]

            if word in (
                self.BOS,
                self.EOS,
                self.PAD
            ):
                continue

            words.append(word)

        return " ".join(words)

    def save(self, path):

        with open(path, "w", encoding="utf-8") as f:

            json.dump(
                self.word_to_id,
                f,
                indent=4
            )

    def load(self, path):

        with open(path, "r", encoding="utf-8") as f:

            self.word_to_id = json.load(f)

        self.id_to_word = {
            int(v): k
            for k, v in self.word_to_id.items()
        }