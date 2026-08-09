import json

from tokenizer.tokenizer import AethyxTokenizer
from tokenizer.train_tokenizer import train_tokenizer
from dataset.dataset import AethyxDataset
import pytest


def test_v2_tokenizer_preserves_case_accents_and_structural_tokens(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("India भारत Café CASE <SYSTEM> instruction\n" * 20), encoding="utf-8"
    )
    output = tmp_path / "tokenizer_v2.json"
    train_tokenizer(corpus, output, vocab_size=300)
    tokenizer = AethyxTokenizer(output)

    assert tokenizer.tokenizer.normalizer.normalize_str("Café CASE") == "Café CASE"
    assert tokenizer.token_to_id("<SYSTEM>") is not None
    sample = "India भारत Café CASE"
    assert tokenizer.decode(tokenizer.encode(sample)) == sample
    metadata = json.loads((tmp_path / "tokenizer_v2_metadata.json").read_text())
    assert metadata["vocab_size"] == tokenizer.vocab_size
    assert metadata["normalizer"]["preserves_case_and_accents"] is True


def test_token_cache_rejects_a_different_tokenizer(tmp_path):
    first_corpus = tmp_path / "first_corpus.txt"
    second_corpus = tmp_path / "second_corpus.txt"
    first_corpus.write_text("alpha beta gamma\n" * 20, encoding="utf-8")
    second_corpus.write_text("भारत delta epsilon\n" * 20, encoding="utf-8")
    first_tokenizer = tmp_path / "first.json"
    second_tokenizer = tmp_path / "second.json"
    train_tokenizer(first_corpus, first_tokenizer, vocab_size=280)
    train_tokenizer(second_corpus, second_tokenizer, vocab_size=280)
    data = tmp_path / "training.txt"
    data.write_text("one document\n\nanother document", encoding="utf-8")
    AethyxDataset(data, context_length=2, tokenizer_path=first_tokenizer)
    with pytest.raises(ValueError, match="different tokenizer"):
        AethyxDataset(data, context_length=2, tokenizer_path=second_tokenizer)
