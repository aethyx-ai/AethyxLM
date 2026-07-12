"""
AethyxLM

Module:
bpe_trainer.py

Purpose:
Train a Byte Pair Encoding (BPE) tokenizer from raw text.

Author:
Aethyx Labs

Version:
v1.0
"""

import json
import re
from collections import Counter


class BPETrainer:

    def __init__(self):

        self.vocab = Counter()
        self.merges = []

    def preprocess(self, text):

        text = text.lower()

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def build_initial_vocab(self, text):

        text = self.preprocess(text)

        words = text.split()

        self.vocab = Counter()

        for word in words:

            chars = " ".join(list(word)) + " </w>"

            self.vocab[chars] += 1

    def get_pair_counts(self):

        pairs = Counter()

        for word, freq in self.vocab.items():

            symbols = word.split()

            for i in range(len(symbols) - 1):

                pairs[(symbols[i], symbols[i + 1])] += freq

        return pairs

    def merge_pair(self, pair):

        pattern = re.escape(" ".join(pair))

        replacement = "".join(pair)

        new_vocab = Counter()

        for word, freq in self.vocab.items():

            new_word = re.sub(
                pattern,
                replacement,
                word
            )

            new_vocab[new_word] += freq

        self.vocab = new_vocab

    def train(self, text, num_merges=100):

        self.build_initial_vocab(text)

        for step in range(num_merges):

            pairs = self.get_pair_counts()

            if not pairs:

                break

            best_pair = max(
                pairs,
                key=pairs.get
            )

            self.merge_pair(best_pair)

            self.merges.append(best_pair)

            print(
                f"[{step+1}/{num_merges}] "
                f"Merged {best_pair}"
            )

    def save_merges(self, path):

        with open(path, "w") as f:

            json.dump(
                self.merges,
                f,
                indent=4
            )

    def save_vocab(self, path):

        vocab = {}

        idx = 0

        for word in self.vocab:

            for token in word.split():

                if token not in vocab:

                    vocab[token] = idx

                    idx += 1

        with open(path, "w") as f:

            json.dump(
                vocab,
                f,
                indent=4
            )