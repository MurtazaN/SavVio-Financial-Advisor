"""
Lightweight guardrails: LLM text must not contradict the authoritative signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List

from llm.response_generator import RecommendationContext


@dataclass
class GuardrailResult:
    passed: bool
    violations: List[str] = field(default_factory=list)


# Phrases that contradict GREEN / YELLOW / RED if they appear in wrong contexts
_RED_BLOCKERS_ON_GREEN = [
    r"\b(do not buy|don't buy|should not buy|must not buy|cannot afford|can't afford|avoid this purchase|strongly advise against)\b",
    r"\bnot recommended\b.*\b(right now|now)\b",
]
_GREEN_BLOCKERS_ON_RED = [
    r"\b(go ahead and buy|safe to buy|you should buy|definitely buy|great time to buy)\b",
]


def check_response(
    text: str,
    recommendation_color: str,
    context: RecommendationContext,
) -> GuardrailResult:
    """
    Verify the explanation does not obviously contradict the deterministic color.

    This is a heuristic layer — not a substitute for NeMo / full policy engines.
    """
    violations: List[str] = []
    if not text or not str(text).strip():
        violations.append("empty_response")
        return GuardrailResult(passed=False, violations=violations)

    t = text.lower()
    color = (recommendation_color or "").upper()

    if color == "GREEN":
        for pat in _RED_BLOCKERS_ON_GREEN:
            if re.search(pat, t, re.IGNORECASE):
                violations.append(f"contradicts_green:{pat}")

    if color == "RED":
        for pat in _GREEN_BLOCKERS_ON_RED:
            if re.search(pat, t, re.IGNORECASE):
                violations.append(f"contradicts_red:{pat}")

    # YELLOW: avoid extreme opposite language
    if color == "YELLOW":
        if re.search(r"\b(do not buy|don't buy)\b", t) and re.search(
            r"\b(go ahead|safe to buy|buy it now)\b", t
        ):
            violations.append("mixed_signals_yellow")

    return GuardrailResult(passed=len(violations) == 0, violations=violations)
