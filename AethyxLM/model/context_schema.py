"""Canonical tensor contract for local context-compiler output."""

from dataclasses import dataclass
from enum import IntEnum

import torch


class ContextType(IntEnum):
    TEXT = 0
    SYSTEM = 1
    CONVERSATION = 2
    TOOL_SCHEMA = 3
    TOOL_RESULT = 4
    DOCUMENT = 5
    ENTITY = 6
    RELATION = 7
    TEMPORAL = 8
    METADATA = 9
    MEMORY = 10
    VISUAL = 11


@dataclass(frozen=True)
class CompiledContextBatch:
    features: torch.Tensor
    type_ids: torch.Tensor
    attention_mask: torch.Tensor
    source_ids: torch.Tensor
    protected_mask: torch.Tensor

    def __post_init__(self):
        if self.features.ndim != 3:
            raise ValueError("features must have shape (batch, items, feature_dim)")
        expected = self.features.shape[:2]
        for name in ("type_ids", "attention_mask", "source_ids", "protected_mask"):
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")

    def to(self, device):
        return CompiledContextBatch(
            self.features.to(device),
            self.type_ids.to(device),
            self.attention_mask.to(device),
            self.source_ids.to(device),
            self.protected_mask.to(device),
        )


def select_context_budget(
    batch: CompiledContextBatch,
    priorities: torch.Tensor,
    budget: int,
) -> CompiledContextBatch:
    """Select a fixed item budget while always retaining protected items.

    Raises instead of silently discarding protected system/tool/exact-value
    material when the requested budget cannot contain it.
    """
    if priorities.shape != batch.features.shape[:2]:
        raise ValueError("priorities must match the context item dimensions")
    if budget <= 0 or budget > batch.features.size(1):
        raise ValueError("budget must be within the available item count")
    selected_rows = []
    for row in range(batch.features.size(0)):
        valid = batch.attention_mask[row].bool()
        protected = batch.protected_mask[row].bool() & valid
        if int(protected.sum()) > budget:
            raise ValueError("budget is smaller than the protected context set")
        adjusted = priorities[row].float().clone()
        adjusted[~valid] = float("-inf")
        adjusted[protected] = float("inf")
        selected_rows.append(torch.topk(adjusted, budget).indices)
    indices = torch.stack(selected_rows)

    def gather(value):
        if value.ndim == 3:
            expanded = indices.unsqueeze(-1).expand(-1, -1, value.size(-1))
            return torch.gather(value, 1, expanded)
        return torch.gather(value, 1, indices)

    return CompiledContextBatch(
        features=gather(batch.features),
        type_ids=gather(batch.type_ids),
        attention_mask=gather(batch.attention_mask),
        source_ids=gather(batch.source_ids),
        protected_mask=gather(batch.protected_mask),
    )
