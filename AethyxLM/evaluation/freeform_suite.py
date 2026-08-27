"""Free-generation capability evaluation without supplied answer choices."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from inference.generation import (
    format_inference_prompt,
    generate_text,
    sampling_for_decoding,
    stop_strings_for_mode,
)


@dataclass(frozen=True)
class FreeFormCase:
    category: str
    base_prompt: str
    question: str
    answers: tuple[str, ...]


FREEFORM_CASES = (
    FreeFormCase("knowledge", "The capital of India is", "What is the capital of India?", ("New Delhi", "Delhi")),
    FreeFormCase("knowledge", "The currency of Japan is the", "What is the currency of Japan?", ("yen", "Japanese yen")),
    FreeFormCase("knowledge", "Hamlet was written by", "Who wrote Hamlet?", ("William Shakespeare", "Shakespeare")),
    FreeFormCase("knowledge", "The largest planet in the Solar System is", "What is the largest planet in the Solar System?", ("Jupiter",)),
    FreeFormCase("science", "Water freezes at", "At what temperature Celsius does water freeze?", ("0 degrees Celsius", "0°C", "zero degrees Celsius")),
    FreeFormCase("science", "The chemical symbol for gold is", "What is the chemical symbol for gold?", ("Au",)),
    FreeFormCase("science", "The organ that pumps blood through the body is the", "Which organ pumps blood through the body?", ("heart",)),
    FreeFormCase("science", "Plants absorb this gas during photosynthesis:", "Which gas do plants absorb during photosynthesis?", ("carbon dioxide", "CO2")),
    FreeFormCase("math", "17 multiplied by 23 equals", "What is 17 multiplied by 23?", ("391",)),
    FreeFormCase("math", "The square root of 144 is", "What is the square root of 144?", ("12", "twelve")),
    FreeFormCase("math", "If x + 7 = 19, then x =", "Solve x + 7 = 19. What is x?", ("12", "x = 12")),
    FreeFormCase("math", "Three quarters written as a decimal is", "Write three quarters as a decimal.", ("0.75",)),
    FreeFormCase("code", "In Python, a function is declared with the keyword", "Which keyword declares a function in Python?", ("def",)),
    FreeFormCase("code", "In JavaScript, strict equality is written as", "How is strict equality written in JavaScript?", ("===",)),
    FreeFormCase("code", "The SQL command used to read rows is", "Which SQL command reads rows from a table?", ("SELECT",)),
    FreeFormCase("code", "A valid HTML hyperlink element begins with", "Which HTML tag creates a hyperlink?", ("<a>", "a")),
    FreeFormCase("language", "The opposite of ancient is", "What is the opposite of ancient?", ("modern",)),
    FreeFormCase("language", "Complete the phrase: better late than", "Complete: better late than ___.", ("never",)),
    FreeFormCase("indic", "भारत की राजधानी है", "भारत की राजधानी क्या है?", ("नई दिल्ली", "दिल्ली")),
    FreeFormCase("indic", "हिंदी में 'water' को कहते हैं", "हिंदी में water को क्या कहते हैं?", ("पानी", "जल")),
    FreeFormCase("indic", "বাংলাদেশের রাজধানী হলো", "বাংলাদেশের রাজধানী কী?", ("ঢাকা",)),
    FreeFormCase("indic", "தமிழ்நாட்டின் தலைநகரம்", "தமிழ்நாட்டின் தலைநகரம் எது?", ("சென்னை",)),
    FreeFormCase("indic", "తెలంగాణ రాజధాని", "తెలంగాణ రాజధాని ఏది?", ("హైదరాబాద్", "హైదరాబాదు")),
)


def normalize_answer(text: str) -> str:
    """Unicode-aware normalization used for transparent exact/prefix scoring."""
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    text = re.sub(r"[\s\u00a0]+", " ", text)
    return text.strip(" \t\r\n.,;:!?\"'`()[]{}")


def score_generated_answer(generation: str, answers: Sequence[str]) -> dict:
    """Score strict equality, answer-prefix, and diagnostic containment separately."""
    normalized = normalize_answer(generation)
    normalized_answers = [normalize_answer(answer) for answer in answers]
    exact = any(normalized == answer for answer in normalized_answers)
    prefix = any(
        normalized == answer
        or normalized.startswith(answer + " ")
        or normalized.startswith(answer + ",")
        for answer in normalized_answers
        if answer
    )
    contains = any(answer in normalized for answer in normalized_answers if answer)
    return {
        "exact_match": exact,
        "answer_prefix_match": prefix,
        "answer_contained": contains,
        "normalized_generation": normalized,
    }


def format_case_prompt(case: FreeFormCase, prompt_mode: str) -> str:
    text = case.base_prompt if prompt_mode == "base" else case.question
    return format_inference_prompt(text, prompt_mode)


@torch.no_grad()
def evaluate_freeform_cases(
    model,
    tokenizer,
    cases: Iterable[FreeFormCase] = FREEFORM_CASES,
    *,
    prompt_mode: str = "base",
    max_new_tokens: int = 24,
    seeds: Sequence[int] = (42, 43, 44),
) -> dict:
    """Run greedy plus seeded sampled generations and report each score honestly."""
    cases = tuple(cases)
    runs = [("greedy", None), *(("sample", seed) for seed in seeds)]
    details = []
    totals: dict[str, list[int]] = {}

    for case in cases:
        prompt = format_case_prompt(case, prompt_mode)
        for decoding, seed in runs:
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            sampling = sampling_for_decoding(
                "greedy" if decoding == "greedy" else "sampled",
                max_new_tokens=max_new_tokens,
            )
            result = generate_text(
                model,
                tokenizer,
                prompt,
                sampling=sampling,
                stop_strings=stop_strings_for_mode(prompt_mode),
            )
            scores = score_generated_answer(result.text, case.answers)
            key = f"{decoding}:{case.category}"
            bucket = totals.setdefault(key, [0, 0, 0, 0])
            bucket[0] += int(scores["exact_match"])
            bucket[1] += int(scores["answer_prefix_match"])
            bucket[2] += int(scores["answer_contained"])
            bucket[3] += 1
            details.append(
                {
                    "category": case.category,
                    "prompt_mode": prompt_mode,
                    "prompt": prompt,
                    "accepted_answers": list(case.answers),
                    "decoding": decoding,
                    "seed": seed,
                    "generation": result.text,
                    "finish_reason": result.finish_reason,
                    **scores,
                }
            )

    def summary(rows: list[dict]) -> dict:
        count = len(rows)
        return {
            "runs": count,
            "exact_match": sum(row["exact_match"] for row in rows) / max(count, 1),
            "answer_prefix_match": sum(row["answer_prefix_match"] for row in rows) / max(count, 1),
            "answer_contained": sum(row["answer_contained"] for row in rows) / max(count, 1),
        }

    greedy = [row for row in details if row["decoding"] == "greedy"]
    sampled = [row for row in details if row["decoding"] == "sample"]
    return {
        "evaluation_type": "free generation; no candidate answers supplied to model",
        "prompt_mode": prompt_mode,
        "cases": len(cases),
        "seeds": list(seeds),
        "greedy": summary(greedy),
        "sampled": summary(sampled),
        "by_decoding_and_category": {
            key: {
                "exact_match": values[0] / values[3],
                "answer_prefix_match": values[1] / values[3],
                "answer_contained": values[2] / values[3],
                "runs": values[3],
            }
            for key, values in totals.items()
        },
        "details": details,
        "warning": (
            "This compact project diagnostic is not a standardized benchmark. "
            "Prefix match is the primary completion metric; containment is diagnostic only."
        ),
    }
