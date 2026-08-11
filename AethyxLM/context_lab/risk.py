"""Conservative guards for information that must remain exact text."""

import re
from dataclasses import dataclass

from context_lab.schema import ContextItem


PRECISION_PATTERNS = (
    re.compile(r"\b[0-9a-fA-F]{16,}\b"),
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}\b"),
    re.compile(r"\b(?:sk|api|token|secret|password)[-_A-Za-z0-9]{8,}\b", re.I),
    re.compile(r"https?://\S+"),
    re.compile(r"(?:[A-Za-z]:\\|/)[^\s]+"),
    re.compile(r"\b\d{8,}\b"),
)


@dataclass(frozen=True)
class RiskAssessment:
    score: float
    reasons: tuple[str, ...]
    must_preserve: bool


def assess_verbatim_risk(item: ContextItem) -> RiskAssessment:
    reasons = []
    score = 0.0
    if item.protected:
        reasons.append("caller_protected")
        score = 1.0
    if item.kind in {"system", "tool_schema", "agent_state"}:
        reasons.append(f"structural:{item.kind}")
        score = max(score, 0.85)
    if "```" in item.text:
        reasons.append("code_block")
        score = max(score, 0.8)
    matched = sum(bool(pattern.search(item.text)) for pattern in PRECISION_PATTERNS)
    if matched:
        reasons.append(f"precision_patterns:{matched}")
        score = max(score, min(1.0, 0.65 + 0.1 * matched))
    if item.recency >= 2 and item.kind == "conversation":
        reasons.append("recent_conversation")
        score = max(score, 0.9)
    return RiskAssessment(score, tuple(reasons), score >= 0.75)


def extract_precision_facts(text: str) -> tuple[str, ...]:
    facts = []
    for pattern in PRECISION_PATTERNS:
        facts.extend(match.group(0) for match in pattern.finditer(text))
    return tuple(dict.fromkeys(facts))

