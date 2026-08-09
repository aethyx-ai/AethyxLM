"""Measure tokenizer fertility and reversible normalization by language sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


def load_documents(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    documents = []
    for part in text.split("<DOC>\n")[1:]:
        document = part.rsplit("\n<EOS>", 1)[0].strip()
        if document:
            documents.append(document)
    return documents


def evaluate(tokenizer: Tokenizer, documents: list[str]) -> dict:
    token_count = 0
    unknown_count = 0
    characters = 0
    utf8_bytes = 0
    exact_roundtrips = 0
    unknown_id = tokenizer.token_to_id("<UNK>")
    for document in documents:
        encoding = tokenizer.encode(document)
        normalized = tokenizer.normalizer.normalize_str(document)
        decoded = tokenizer.decode(encoding.ids)
        token_count += len(encoding.ids)
        unknown_count += sum(token_id == unknown_id for token_id in encoding.ids)
        characters += len(normalized)
        utf8_bytes += len(normalized.encode("utf-8"))
        exact_roundtrips += int(decoded == normalized)
    return {
        "documents": len(documents),
        "tokens": token_count,
        "characters_per_token": characters / max(1, token_count),
        "utf8_bytes_per_token": utf8_bytes / max(1, token_count),
        "unknown_token_rate": unknown_count / max(1, token_count),
        "normalized_roundtrip_rate": exact_roundtrips / max(1, len(documents)),
    }


def main(args: argparse.Namespace) -> None:
    corpus = Path(args.corpus)
    corpus_metadata = json.loads(
        Path(args.corpus_metadata).read_text(encoding="utf-8")
    )
    documents = load_documents(corpus)
    tokenizers = {
        name: Tokenizer.from_file(str(path))
        for name, path in (
            ("legacy", Path(args.legacy_tokenizer)),
            ("v2", Path(args.v2_tokenizer)),
        )
    }
    result = {
        "evaluation_type": "in-sample tokenizer engineering diagnostic",
        "warning": "These samples trained tokenizer v2; results are not held-out quality estimates.",
        "corpus_sha256": corpus_metadata["sha256"],
        "tokenizers": {},
    }

    start = 0
    language_documents = {}
    for language, source in corpus_metadata["sources"].items():
        count = source["documents"]
        candidates = documents[start : start + count]
        start += count
        # Evenly cover the bounded per-language segment without evaluating it all.
        if len(candidates) > args.documents_per_language:
            indices = [
                round(i * (len(candidates) - 1) / (args.documents_per_language - 1))
                for i in range(args.documents_per_language)
            ]
            candidates = [candidates[index] for index in indices]
        language_documents[language] = candidates

    for name, tokenizer in tokenizers.items():
        per_language = {
            language: evaluate(tokenizer, samples)
            for language, samples in language_documents.items()
        }
        all_samples = [sample for samples in language_documents.values() for sample in samples]
        result["tokenizers"][name] = {
            "vocab_size": tokenizer.get_vocab_size(),
            "aggregate": evaluate(tokenizer, all_samples),
            "per_language": per_language,
        }

    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({name: value["aggregate"] for name, value in result["tokenizers"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="tokenizer/data/multilingual_corpus.txt")
    parser.add_argument(
        "--corpus-metadata",
        default="tokenizer/data/multilingual_corpus.txt.metadata.json",
    )
    parser.add_argument("--legacy-tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--v2-tokenizer", default="tokenizer/tokenizer_v2.json")
    parser.add_argument("--documents-per-language", type=int, default=50)
    parser.add_argument("--output", default="tokenizer/tokenizer_v2_evaluation.json")
    parsed = parser.parse_args()
    if parsed.documents_per_language < 2:
        parser.error("--documents-per-language must be at least 2")
    main(parsed)
