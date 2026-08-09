"""Query-aware retrieval over locally compiled context features."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RetrievalResult:
    features: torch.Tensor
    type_ids: torch.Tensor
    source_ids: torch.Tensor
    attention_mask: torch.Tensor
    scores: torch.Tensor


class ContextMemoryBank:
    """Immutable batched context store with deterministic cosine retrieval.

    Source IDs are carried through retrieval so a caller can request the exact
    original material when a compressed representation is insufficient.
    """

    def __init__(
        self,
        features: torch.Tensor,
        type_ids: Optional[torch.Tensor] = None,
        source_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if features.ndim != 3:
            raise ValueError("memory features must have shape (batch, items, dim)")
        batch, items, _ = features.shape
        expected = (batch, items)
        self.features = features
        self.type_ids = (
            torch.zeros(expected, device=features.device, dtype=torch.long)
            if type_ids is None
            else type_ids
        )
        self.source_ids = (
            torch.arange(items, device=features.device)
            .unsqueeze(0)
            .expand(batch, -1)
            if source_ids is None
            else source_ids
        )
        self.attention_mask = (
            torch.ones(expected, device=features.device, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.bool()
        )
        for name, value in (
            ("type_ids", self.type_ids),
            ("source_ids", self.source_ids),
            ("attention_mask", self.attention_mask),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if not self.attention_mask.any(dim=1).all():
            raise ValueError("every memory batch row must contain at least one valid item")

    def retrieve(self, query: torch.Tensor, top_k: int) -> RetrievalResult:
        if query.ndim == 3 and query.size(1) == 1:
            query = query[:, 0]
        if query.ndim != 2 or query.shape != (
            self.features.size(0),
            self.features.size(2),
        ):
            raise ValueError("query must have shape (batch, feature_dim)")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        top_k = min(top_k, self.features.size(1))

        normalized_memory = F.normalize(self.features.float(), dim=-1)
        normalized_query = F.normalize(query.float(), dim=-1)
        scores = torch.einsum("bnd,bd->bn", normalized_memory, normalized_query)
        scores = scores.masked_fill(~self.attention_mask, float("-inf"))
        top_scores, indices = torch.topk(scores, top_k, dim=-1)
        valid = torch.isfinite(top_scores)

        def gather(values: torch.Tensor):
            if values.ndim == 3:
                expanded = indices.unsqueeze(-1).expand(-1, -1, values.size(-1))
                return torch.gather(values, 1, expanded)
            return torch.gather(values, 1, indices)

        return RetrievalResult(
            features=gather(self.features),
            type_ids=gather(self.type_ids),
            source_ids=gather(self.source_ids),
            attention_mask=valid,
            scores=top_scores.masked_fill(~valid, 0.0),
        )
