"""Compare tokenizer v3 with the production v2 tokenizer on held-out data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


def load_documents(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    documents = []
    for part in text.split("<DOC>\n")[1:]:
        document = part.rsplit("\n<EOS>", 1)[0]
        if document:
            documents.append(document)
    return documents


def metrics(tokenizer: Tokenizer, documents: list[str]) -> dict:
    tokens = characters = utf8_bytes = unknown = roundtrips = 0
    unknown_id = tokenizer.token_to_id("<UNK>")
    for document in documents:
        encoding = tokenizer.encode(document)
        normalized = tokenizer.normalizer.normalize_str(document)
        decoded = tokenizer.decode(encoding.ids)
        tokens += len(encoding.ids)
        characters += len(normalized)
        utf8_bytes += len(normalized.encode("utf-8"))
        unknown += sum(token == unknown_id for token in encoding.ids)
        roundtrips += int(decoded == normalized)
    return {
        "documents": len(documents),
        "tokens": tokens,
        "characters_per_token": characters / max(1, tokens),
        "utf8_bytes_per_token": utf8_bytes / max(1, tokens),
        "unknown_token_rate": unknown / max(1, tokens),
        "normalized_roundtrip_rate": roundtrips / max(1, len(documents)),
    }


def main(args: argparse.Namespace) -> dict:
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    documents = load_documents(Path(args.corpus))
    expected = sum(source["heldout_documents"] for source in metadata["sources"])
    if len(documents) != expected:
        raise ValueError(f"Held-out document count is {len(documents)}, expected {expected}")
    tokenizers = {
        "v2_32k": Tokenizer.from_file(args.baseline),
        "v3_48k": Tokenizer.from_file(args.candidate),
    }
    result = {
        "evaluation_type": "held-out tokenizer fertility and reversibility",
        "heldout_sha256": metadata["heldout"]["sha256"],
        "tokenizers": {},
    }
    for name, tokenizer in tokenizers.items():
        start = 0
        per_source = {}
        for source in metadata["sources"]:
            count = source["heldout_documents"]
            samples = documents[start : start + count]
            start += count
            per_source[source["name"]] = metrics(tokenizer, samples)
        result["tokenizers"][name] = {
            "vocab_size": tokenizer.get_vocab_size(),
            "aggregate": metrics(tokenizer, documents),
            "per_source": per_source,
        }
    baseline_tokens = result["tokenizers"]["v2_32k"]["aggregate"]["tokens"]
    candidate_tokens = result["tokenizers"]["v3_48k"]["aggregate"]["tokens"]
    result["v3_token_reduction_vs_v2"] = 1.0 - candidate_tokens / max(1, baseline_tokens)
    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "v3_token_reduction_vs_v2": result["v3_token_reduction_vs_v2"],
        "v2": result["tokenizers"]["v2_32k"]["aggregate"],
        "v3": result["tokenizers"]["v3_48k"]["aggregate"],
    }, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="tokenizer/data/tokenizer_v3_heldout.txt")
    parser.add_argument("--metadata", default="tokenizer/data/tokenizer_v3_corpus.txt.metadata.json")
    parser.add_argument("--baseline", default="tokenizer/tokenizer.json")
    parser.add_argument("--candidate", default="tokenizer/tokenizer_v3_48k.json")
    parser.add_argument("--output", default="tokenizer/tokenizer_v3_48k_evaluation.json")
    main(parser.parse_args())
