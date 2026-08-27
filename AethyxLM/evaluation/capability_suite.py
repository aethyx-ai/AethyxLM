"""Deterministic capability checks suitable for base language-model checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class ChoiceCase:
    category: str
    prompt: str
    choices: tuple[str, ...]
    answer_index: int


CHOICE_CASES = (
    ChoiceCase("knowledge", "The capital of India is", (" Mumbai", " New Delhi", " Kolkata", " Chennai"), 1),
    ChoiceCase("knowledge", "The largest planet in the Solar System is", (" Mars", " Venus", " Jupiter", " Mercury"), 2),
    ChoiceCase("knowledge", "Water freezes at", (" 0 degrees Celsius", " 50 degrees Celsius", " 100 degrees Celsius", " -100 degrees Celsius"), 0),
    ChoiceCase("knowledge", "The currency of Japan is the", (" dollar", " euro", " yen", " rupee"), 2),
    ChoiceCase("knowledge", "The author of Hamlet was", (" Charles Dickens", " William Shakespeare", " Jane Austen", " George Orwell"), 1),
    ChoiceCase("science", "Plants absorb this gas during photosynthesis:", (" oxygen", " nitrogen", " carbon dioxide", " helium"), 2),
    ChoiceCase("science", "The chemical symbol for gold is", (" Ag", " Au", " Fe", " Gd"), 1),
    ChoiceCase("science", "The organ that pumps blood through the body is the", (" liver", " kidney", " heart", " lung"), 2),
    ChoiceCase("science", "A force measured in newtons is", (" mass", " temperature", " force", " voltage"), 2),
    ChoiceCase("science", "The Earth completes one orbit around the Sun in approximately", (" one day", " one month", " one year", " ten years"), 2),
    ChoiceCase("math", "17 multiplied by 23 equals", (" 381", " 391", " 401", " 411"), 1),
    ChoiceCase("math", "The square root of 144 is", (" 10", " 11", " 12", " 14"), 2),
    ChoiceCase("math", "If x + 7 = 19, then x =", (" 10", " 11", " 12", " 13"), 2),
    ChoiceCase("math", "Three quarters written as a decimal is", (" 0.25", " 0.5", " 0.75", " 1.25"), 2),
    ChoiceCase("math", "The next prime number after 11 is", (" 12", " 13", " 15", " 17"), 1),
    ChoiceCase("code", "In Python, a function is declared with the keyword", (" function", " def", " fn", " lambda"), 1),
    ChoiceCase("code", "In JavaScript, strict equality is written as", (" =", " ==", " ===", " !="), 2),
    ChoiceCase("code", "The SQL command used to read rows is", (" SELECT", " INSERT", " UPDATE", " DROP"), 0),
    ChoiceCase("code", "A valid HTML hyperlink element begins with", (" <p>", " <a>", " <img>", " <div>"), 1),
    ChoiceCase("code", "In Rust, an immutable variable is commonly introduced with", (" let", " var", " consteval", " auto"), 0),
    ChoiceCase("language", "Birds can fly, but fish can", (" swim", " read", " drive", " write"), 0),
    ChoiceCase("language", "She went to the market because she needed to buy", (" groceries", " thunder", " silence", " distance"), 0),
    ChoiceCase("language", "The opposite of ancient is", (" old", " modern", " historic", " antique"), 1),
    ChoiceCase("language", "A person who teaches students is a", (" teacher", " pilot", " carpenter", " sailor"), 0),
    ChoiceCase("language", "Complete the phrase: better late than", (" early", " never", " always", " yesterday"), 1),
    ChoiceCase("indic", "भारत की राजधानी है", (" मुंबई", " नई दिल्ली", " चेन्नई", " कोलकाता"), 1),
    ChoiceCase("indic", "हिंदी में 'water' को कहते हैं", (" आग", " पानी", " हवा", " धरती"), 1),
    ChoiceCase("indic", "বাংলাদেশের রাজধানী হলো", (" ঢাকা", " কলকাতা", " চট্টগ্রাম", " দিল্লি"), 0),
    ChoiceCase("indic", "தமிழ்நாட்டின் தலைநகரம்", (" மதுரை", " கோயம்புத்தூர்", " சென்னை", " சேலம்"), 2),
    ChoiceCase("indic", "తెలంగాణ రాజధాని", (" హైదరాబాద్", " చెన్నై", " ముంబై", " పుణె"), 0),
)


@torch.no_grad()
def conditional_mean_logprob(model, tokenizer, prompt: str, completion: str) -> tuple[float, bool]:
    """Return length-normalized log probability for a completion."""
    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(completion)
    if not prompt_ids or not completion_ids:
        return float("-inf"), False
    max_prompt = max(1, model.context_length - len(completion_ids))
    truncated = len(prompt_ids) > max_prompt
    prompt_ids = prompt_ids[-max_prompt:]
    sequence = prompt_ids + completion_ids
    device = next(model.parameters()).device
    inputs = torch.tensor([sequence[:-1]], device=device)
    targets = torch.tensor(completion_ids, device=device)
    logits = model(inputs)[0]
    start = len(prompt_ids) - 1
    scored = logits[start : start + len(completion_ids)]
    log_probs = torch.log_softmax(scored.float(), dim=-1)
    values = log_probs.gather(1, targets[:, None]).squeeze(1)
    return float(values.mean().item()), truncated


def choose_completion(model, tokenizer, prompt: str, choices: Sequence[str]) -> dict:
    scores = []
    truncated = False
    for choice in choices:
        score, was_truncated = conditional_mean_logprob(model, tokenizer, prompt, choice)
        scores.append(score)
        truncated = truncated or was_truncated
    selected = max(range(len(scores)), key=scores.__getitem__)
    return {"selected_index": selected, "scores": scores, "input_truncated": truncated}


def evaluate_choice_cases(
    model,
    tokenizer,
    cases: Iterable[ChoiceCase] = CHOICE_CASES,
) -> dict:
    rows = []
    totals: dict[str, list[int]] = {}
    for case in cases:
        result = choose_completion(model, tokenizer, case.prompt, case.choices)
        correct = result["selected_index"] == case.answer_index
        bucket = totals.setdefault(case.category, [0, 0])
        bucket[0] += int(correct)
        bucket[1] += 1
        rows.append(
            {
                "category": case.category,
                "prompt": case.prompt,
                "choices": list(case.choices),
                "answer_index": case.answer_index,
                "selected_index": result["selected_index"],
                "scores": result["scores"],
                "correct": correct,
            }
        )
    correct = sum(int(row["correct"]) for row in rows)
    return {
        "accuracy": correct / max(len(rows), 1),
        "correct": correct,
        "cases": len(rows),
        "by_category": {
            name: {"accuracy": values[0] / values[1], "correct": values[0], "cases": values[1]}
            for name, values in totals.items()
        },
        "details": rows,
        "warning": "This is a compact diagnostic suite, not a standardized benchmark.",
    }
