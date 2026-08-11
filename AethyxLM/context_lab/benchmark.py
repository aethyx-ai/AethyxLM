"""Representation benchmark that scores reduction only with task retention."""

from __future__ import annotations

import json
from dataclasses import dataclass

from context_lab.compiler import LocalContextCompiler
from context_lab.risk import extract_precision_facts
from context_lab.schema import ContextRequest


@dataclass(frozen=True)
class BenchmarkCase:
    request: ContextRequest
    relevant_source_ids: tuple[str, ...]
    exact_values: tuple[str, ...] = ()


def benchmark_compiler(
    compiler: LocalContextCompiler,
    tokenizer,
    cases: list[BenchmarkCase],
):
    if not cases:
        raise ValueError("at least one benchmark case is required")
    baseline_tokens = 0
    compiled_tokens = 0
    relevant_total = 0
    relevant_recalled = 0
    exact_total = 0
    exact_retained = 0
    visual_pages = 0
    details = []
    for case in cases:
        compiled = compiler.compile(case.request)
        source_text = case.request.query + "\n" + "\n".join(
            item.text for item in case.request.items
        )
        model_payload = compiled.model_payload()
        serialized = json.dumps(
            model_payload, ensure_ascii=False, separators=(",", ":")
        )
        source_token_count = len(tokenizer.encode(source_text))
        compiled_token_count = (
            source_token_count
            if compiled.mode == "raw"
            else len(tokenizer.encode(serialized))
        )
        baseline_tokens += source_token_count
        compiled_tokens += compiled_token_count
        represented = {
            item.source_id for item in compiled.protected_items + compiled.selected_items
        }
        represented.update(
            source_id for page in compiled.visual_pages for source_id in page.source_ids
        )
        relevant = set(case.relevant_source_ids)
        relevant_total += len(relevant)
        relevant_recalled += len(relevant & represented)
        retained_text = "\n".join(
            item.text for item in compiled.protected_items + compiled.selected_items
        )
        expected_exact = case.exact_values or extract_precision_facts(source_text)
        exact_total += len(expected_exact)
        exact_retained += sum(value in retained_text for value in expected_exact)
        visual_pages += len(compiled.visual_pages)
        details.append(
            {
                "request_id": case.request.request_id,
                "selected_mode": compiled.mode,
                "source_tokens": source_token_count,
                "compiled_text_equivalent_tokens": compiled_token_count,
                "audit_manifest_tokens": len(
                    tokenizer.encode(
                        json.dumps(
                            compiled.manifest(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                ),
                "relevant_recall": len(relevant & represented) / max(len(relevant), 1),
                "warnings": compiled.warnings,
            }
        )
    reduction = 1 - compiled_tokens / max(baseline_tokens, 1)
    return {
        "representation": compiler.policy.mode,
        "cases": len(cases),
        "baseline_text_tokens": baseline_tokens,
        "compiled_text_equivalent_tokens": compiled_tokens,
        "text_equivalent_reduction": reduction,
        "relevant_source_recall": relevant_recalled / max(relevant_total, 1),
        "exact_value_retention": exact_retained / max(exact_total, 1),
        "visual_pages": visual_pages,
        "model_task_accuracy": None,
        "warning": (
            "Token reduction is measured on the serialized research envelope. "
            "Visual pages have not been assigned text-token equivalents and no "
            "model accuracy claim is made until a compatible encoder is evaluated."
        ),
        "details": details,
    }
