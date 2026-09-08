"""Small, dependency-free evidence retrieval with stable source identifiers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable


_WORD = re.compile(r"[\w.+#/-]+", re.UNICODE)


def _terms(text: str):
    return [term.casefold() for term in _WORD.findall(text)]


@dataclass(frozen=True)
class EvidencePassage:
    source_id: str
    text: str
    score: float = 0.0


class EvidenceIndex:
    """BM25-style local retriever; passage text is returned verbatim."""

    def __init__(self, passages: Iterable[EvidencePassage]):
        self.passages = tuple(passages)
        if len({item.source_id for item in self.passages}) != len(self.passages):
            raise ValueError("evidence source_id values must be unique")
        self._documents = [_terms(item.text) for item in self.passages]
        self._avg_len = sum(map(len, self._documents)) / max(len(self._documents), 1)
        self._df: dict[str, int] = {}
        for document in self._documents:
            for term in set(document):
                self._df[term] = self._df.get(term, 0) + 1

    def search(self, query: str, limit: int = 4) -> list[EvidencePassage]:
        if limit <= 0:
            return []
        query_terms = set(_terms(query))
        scored = []
        count = len(self.passages)
        for passage, document in zip(self.passages, self._documents):
            frequencies = {term: document.count(term) for term in query_terms}
            score = 0.0
            for term, frequency in frequencies.items():
                if not frequency:
                    continue
                inverse = math.log(1 + (count - self._df.get(term, 0) + 0.5) / (self._df.get(term, 0) + 0.5))
                denominator = frequency + 1.2 * (0.25 + 0.75 * len(document) / max(self._avg_len, 1))
                score += inverse * frequency * 2.2 / denominator
            if score > 0:
                scored.append(EvidencePassage(passage.source_id, passage.text, score))
        return sorted(scored, key=lambda item: (-item.score, item.source_id))[:limit]


def format_evidence(passages, tokenizer=None, max_tokens: int | None = None) -> str:
    """Format whole passages within a budget so identifiers and numbers stay intact."""
    blocks = [f"[SOURCE {item.source_id}]\n{item.text}" for item in passages]
    if tokenizer is None or max_tokens is None:
        return "\n\n".join(blocks)
    selected: list[str] = []
    for block in blocks:
        candidate = "\n\n".join((*selected, block))
        if len(tokenizer.encode(candidate)) <= max_tokens:
            selected.append(block)
    return "\n\n".join(selected)
