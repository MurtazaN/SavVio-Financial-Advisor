"""
Parse natural-language purchase queries into intent + product reference.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_price_hint(text: str) -> Optional[float]:
    """Parse a dollar amount like $350 or $1,200.99 from the user message."""
    if not text:
        return None
    m = re.search(r"\$\s*([\d,]+(?:\.\d{1,2})?)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def clean_product_reference(ref: Optional[str]) -> Optional[str]:
    """Remove a leading price so catalog search uses product words (e.g. 'headphones' not '$350 headphones')."""
    if not ref:
        return None
    s = ref.strip()
    s = re.sub(r"^\s*\$\s*[\d,]+(?:\.\d{1,2})?\s+", "", s)
    s = re.sub(r"^\s*[\d,]+(?:\.\d{1,2})?\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) >= 2 else None


@dataclass
class ParsedIntent:
    """Result of parse_user_input."""

    intent: str  # "purchase_query" | "out_of_scope"
    product_reference: Optional[str] = None
    user_context: Optional[str] = None
    # Dollar amount from the query (e.g. $350) — used to match catalog price or hypothetical buy
    price_hint: Optional[float] = None


def _heuristic_parse(query: str) -> ParsedIntent:
    """Regex/heuristic fallback when LLM is unavailable or returns junk."""
    q = (query or "").strip()
    if not q:
        return ParsedIntent(intent="out_of_scope", product_reference=None)

    lower = q.lower()
    # Obvious non-purchase chit-chat
    if re.match(r"^(hi|hello|hey|thanks|thank you|ok|okay)\b", lower) and len(q) < 40:
        return ParsedIntent(intent="out_of_scope", product_reference=None)

    # Extract quoted strings
    m = re.search(r"['\"]([^'\"]{2,80})['\"]", q)
    if m:
        return ParsedIntent(intent="purchase_query", product_reference=m.group(1).strip())

    # "$2,500 Peloton worth it?" — take the product words after the price, stop before " worth " or ? / end.
    # Must run before any pattern that matches the word "worth" in "worth it?" (which would capture "it").
    m = re.search(
        r"\$[\d,]+(?:\.\d{1,2})?\s+([A-Za-z][^?$]*?)(?=\s+worth\b|\s*\?|!|\.|$)",
        q,
        re.IGNORECASE,
    )
    if m:
        ref = m.group(1).strip()
        ref = re.sub(r"\s+", " ", ref)
        if len(ref) >= 2:
            return ParsedIntent(intent="purchase_query", product_reference=ref)

    # "buy a X" / "afford X" — do not use the word "worth" here; it matches "worth it?" and yields "it".
    m = re.search(
        r"(?:buy|afford|purchase|get|considering)\s+(?:a|an|the)?\s*([^?.!]{2,100})",
        q,
        re.IGNORECASE,
    )
    if m:
        ref = m.group(1).strip()
        ref = re.sub(r"\s+", " ", ref)
        # Trim trailing price-only noise
        ref = re.sub(r"\s*for\s*\$.*$", "", ref, flags=re.IGNORECASE)
        if ref:
            return ParsedIntent(intent="purchase_query", product_reference=ref)

    # Dollar amount + product chunk (fallback if no "worth" in the question)
    m = re.search(r"\$[\d,]+(?:\.\d{2})?\s+(.{3,80})", q)
    if m:
        ref = m.group(1).strip()
        ref = re.sub(r"\s+worth(\s+it)?\s*[\?!.]?\s*$", "", ref, flags=re.IGNORECASE)
        ref = re.sub(r"\s+", " ", ref).strip()
        if len(ref) >= 2:
            return ParsedIntent(intent="purchase_query", product_reference=ref)

    # Last resort: strip question words and use remainder
    cleaned = re.sub(
        r"^(should i|can i|is it|is a|would it be|do you think i should)\s+",
        "",
        lower,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\?$", "", cleaned).strip()
    if len(cleaned) >= 3:
        return ParsedIntent(intent="purchase_query", product_reference=cleaned[:120])

    return ParsedIntent(intent="out_of_scope", product_reference=None)


def _parse_llm_json(text: str) -> Optional[dict]:
    """Extract JSON object from LLM output."""
    text = text.strip()
    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                text = p
                break
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_user_input(query: str, llm_provider: Any) -> ParsedIntent:
    """
    Classify intent and extract a product name/reference for catalog lookup.

    Uses LLM when provider supports generate(); falls back to heuristics.
    """
    q = (query or "").strip()
    if not q:
        return ParsedIntent(intent="out_of_scope")

    prompt = f"""Analyze this user message for a financial purchase-advisor app.

Return ONLY valid JSON (no markdown) with keys:
- "intent": either "purchase_query" (user wants to evaluate buying something) or "out_of_scope" (greetings, unrelated topics, general chat).
- "product_reference": short string naming the product or category to search — words only, no leading dollar amounts (null if unclear).
- "price_hint": number if the user gave a price like $350, else null.
- "user_context": optional short note about timing or constraints (or null).

User message: {q!r}
"""

    try:
        raw = llm_provider.generate(prompt, max_tokens=256, temperature=0.1)
        data = _parse_llm_json(raw)
        if data:
            intent = str(data.get("intent", "out_of_scope")).lower()
            if intent not in ("purchase_query", "out_of_scope"):
                intent = "purchase_query" if data.get("product_reference") else "out_of_scope"
            if intent == "out_of_scope":
                return ParsedIntent(intent="out_of_scope")
            pref = data.get("product_reference")
            if isinstance(pref, str):
                pref = pref.strip() or None
                if pref and pref.lower() in ("unknown", "unknown product", "none", "n/a"):
                    pref = None
            else:
                pref = None
            uctx = data.get("user_context")
            if isinstance(uctx, str):
                uctx = uctx.strip() or None
            else:
                uctx = None
            ph = data.get("price_hint")
            price_hint: Optional[float] = None
            if isinstance(ph, (int, float)):
                price_hint = float(ph)
            elif isinstance(ph, str):
                try:
                    price_hint = float(ph.replace(",", "").replace("$", "").strip())
                except ValueError:
                    price_hint = None
            if intent == "purchase_query" and not pref:
                h = _heuristic_parse(q)
                return _merge_price_and_clean(q, h)
            pref = clean_product_reference(pref) or pref
            merged = ParsedIntent(intent=intent, product_reference=pref, user_context=uctx, price_hint=price_hint)
            return _merge_price_and_clean(q, merged)
    except Exception as e:
        logger.debug("LLM intent parse failed, using heuristic: %s", e)

    return _merge_price_and_clean(q, _heuristic_parse(q))


def _merge_price_and_clean(query: str, parsed: ParsedIntent) -> ParsedIntent:
    """Attach price from the raw query; strip $ amounts from product_reference for search."""
    ph = extract_price_hint(query)
    if ph is not None:
        parsed.price_hint = ph
    if parsed.product_reference:
        cleaned = clean_product_reference(parsed.product_reference)
        if cleaned:
            parsed.product_reference = cleaned
    # Heuristic misfires can leave "it" / "this" when the real product follows "$price Brand"
    if (
        parsed.product_reference
        and parsed.product_reference.lower() in ("it", "this", "that")
        and ph is not None
    ):
        m = re.search(
            r"\$[\d,]+(?:\.\d{1,2})?\s+([A-Za-z][^?$]*?)(?=\s+worth\b|\s*\?|!|\.|$)",
            query,
            re.IGNORECASE,
        )
        if m:
            parsed.product_reference = re.sub(r"\s+", " ", m.group(1).strip())
    return parsed
