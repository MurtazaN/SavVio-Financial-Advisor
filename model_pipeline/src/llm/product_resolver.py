"""
Resolve product references to database rows (ILIKE search or by ID).

When the user gives a price (e.g. $350 headphones), we score catalog rows by
name overlap *and* price closeness so we do not match a random cheap item.
If nothing fits, return None so inference can evaluate a hypothetical purchase
at the stated price.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Max relative price gap between user-stated price and catalog row (e.g. 0.35 => ±35%)
_MAX_PRICE_MISMATCH = 0.38


@dataclass
class ProductMatch:
    product_id: str
    product_name: str
    price: float


def resolve_product_by_id(product_id: str, db_engine: Any) -> Optional[ProductMatch]:
    """Exact lookup by products.product_id."""
    if not product_id or db_engine is None:
        return None
    sql = text("""
        SELECT product_id, product_name, price
        FROM products
        WHERE product_id = :pid
        LIMIT 1
    """)
    try:
        with db_engine.connect() as conn:
            row = conn.execute(sql, {"pid": product_id.strip()}).fetchone()
    except Exception as e:
        logger.error("resolve_product_by_id failed: %s", e)
        return None
    if row is None:
        return None
    return ProductMatch(
        product_id=str(row[0]),
        product_name=str(row[1]),
        price=float(row[2] or 0),
    )


def _tokenize(s: str) -> list[str]:
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    return [t for t in s.split() if len(t) > 1]


def _is_noise_token(t: str) -> bool:
    """Skip pure numbers / prices — they widen ILIKE matches incorrectly."""
    if re.match(r"^[\d,]+\.?\d*$", t):
        return True
    if len(t) < 3 and t.isdigit():
        return True
    return False


def _name_overlap_score(ref_tokens: set[str], name: str) -> float:
    name_tokens = set(_tokenize(name))
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & name_tokens) / max(len(ref_tokens), 1)


def _price_score(price: float, hint: Optional[float]) -> float:
    if hint is None or hint <= 0:
        return 1.0
    err = abs(float(price) - hint) / max(hint, 1.0)
    return max(0.0, 1.0 - err / max(_MAX_PRICE_MISMATCH, 0.01))


def resolve_product(
    product_reference: str,
    db_engine: Any,
    price_hint: Optional[float] = None,
    limit_scan: int = 120,
) -> Optional[ProductMatch]:
    """
    Find a product by fuzzy name + optional price alignment.

    If ``price_hint`` is set and every candidate's price is far from it, returns
    None (caller should run a hypothetical evaluation at ``price_hint``).
    """
    if not product_reference or db_engine is None:
        return None

    ref = product_reference.strip()
    if not ref:
        return None

    sql_like = text("""
        SELECT product_id, product_name, price
        FROM products
        WHERE product_name ILIKE :pat
        ORDER BY rating_number DESC NULLS LAST
        LIMIT 8
    """)

    candidates: list[tuple] = []

    try:
        with db_engine.connect() as conn:
            rows = conn.execute(sql_like, {"pat": f"%{ref}%"}).fetchall()
            candidates.extend(rows)

            if not candidates:
                tokens = [t for t in _tokenize(ref) if not _is_noise_token(t)]
                for tok in tokens[:6]:
                    rows = conn.execute(sql_like, {"pat": f"%{tok}%"}).fetchall()
                    candidates.extend(rows[:12])

            if not candidates:
                return None

            seen: set[str] = set()
            unique: list[tuple] = []
            for r in candidates:
                pid = str(r[0])
                if pid not in seen:
                    seen.add(pid)
                    unique.append(r)

            ref_tokens = set(t for t in _tokenize(ref) if not _is_noise_token(t))

            best: Optional[tuple] = None
            best_score = -1.0
            for r in unique[:limit_scan]:
                name = str(r[1])
                price = float(r[2] or 0)
                ns = _name_overlap_score(ref_tokens, name)
                ps = _price_score(price, price_hint)
                if price_hint is not None:
                    combined = 0.55 * ns + 0.45 * ps
                else:
                    combined = ns if ref_tokens else 0.5
                if combined > best_score:
                    best_score = combined
                    best = r

            if best is None:
                return None

            price = float(best[2] or 0)
            best_name = str(best[1])
            name_match = _name_overlap_score(ref_tokens, best_name) if ref_tokens else 1.0

            if price_hint is not None and ref_tokens and name_match < 0.12:
                logger.info(
                    "No catalog row matches product words — hypothetical mode (hint=%.2f)",
                    price_hint,
                )
                return None

            if price_hint is not None and price_hint > 0:
                rel_err = abs(price - price_hint) / price_hint
                if rel_err > _MAX_PRICE_MISMATCH:
                    logger.info(
                        "Rejecting catalog match due to price vs hint: catalog=%.2f hint=%.2f",
                        price,
                        price_hint,
                    )
                    return None

            return ProductMatch(
                product_id=str(best[0]),
                product_name=str(best[1]),
                price=price,
            )
    except Exception as e:
        logger.error("resolve_product failed: %s", e)
        return None
