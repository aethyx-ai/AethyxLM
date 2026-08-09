import torch
import pytest

from model.context_memory import ContextMemoryBank


def test_query_aware_retrieval_preserves_types_and_source_references():
    features = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [-1.0, 0.0]]]
    )
    types = torch.tensor([[2, 3, 4, 5]])
    sources = torch.tensor([[10, 11, 12, 13]])
    memory = ContextMemoryBank(features, types, sources)
    result = memory.retrieve(torch.tensor([[1.0, 0.0]]), top_k=2)
    assert result.source_ids.tolist() == [[10, 12]]
    assert result.type_ids.tolist() == [[2, 4]]
    assert result.attention_mask.all()


def test_masked_items_are_never_returned_as_valid():
    features = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]])
    memory = ContextMemoryBank(
        features, attention_mask=torch.tensor([[False, True, False]])
    )
    result = memory.retrieve(torch.tensor([[1.0, 0.0]]), top_k=3)
    assert result.attention_mask.sum() == 1
    assert result.source_ids[result.attention_mask].item() == 1


def test_entirely_empty_memory_rows_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        ContextMemoryBank(
            torch.randn(1, 3, 4), attention_mask=torch.zeros(1, 3, dtype=torch.bool)
        )
