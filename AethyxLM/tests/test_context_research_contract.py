import pytest
import torch

from model.context_adapter import LatentContextAdapter
from model.context_objectives import ContextPreservationObjective
from model.context_schema import CompiledContextBatch, ContextType, select_context_budget


def compiled_batch():
    return CompiledContextBatch(
        features=torch.randn(2, 6, 12),
        type_ids=torch.tensor(
            [[ContextType.SYSTEM, 5, 5, 5, 5, 5], [1, 2, 3, 4, 5, 6]]
        ),
        attention_mask=torch.ones(2, 6, dtype=torch.bool),
        source_ids=torch.arange(6).repeat(2, 1),
        protected_mask=torch.tensor(
            [[True, False, False, False, False, False], [True, True, False, False, False, False]]
        ),
    )


def test_budget_selection_never_drops_protected_context():
    batch = compiled_batch()
    priorities = torch.rand(2, 6)
    selected = select_context_budget(batch, priorities, budget=3)
    assert selected.protected_mask.sum(1).tolist() == [1, 2]
    assert selected.features.shape == (2, 3, 12)


def test_impossible_protected_budget_fails_explicitly():
    with pytest.raises(ValueError, match="protected"):
        select_context_budget(compiled_batch(), torch.rand(2, 6), budget=1)


def test_preservation_objective_backpropagates_into_adapter():
    adapter = LatentContextAdapter(12, 16, num_latents=4, num_heads=4, depth=1)
    objective = ContextPreservationObjective(16, 12, num_heads=4)
    source = torch.randn(3, 8, 12)
    latents = adapter(source)
    losses = objective(latents, source)
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert adapter.latents.grad is not None
    assert adapter.latents.grad.abs().sum() > 0
