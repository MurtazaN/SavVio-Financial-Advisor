"""DB-backed product catalog search and retrieval."""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)


def list_products(
    db_engine: Any,
    *,
    q: Optional[str],
    price_min: Optional[float],
    price_max: Optional[float],
    limit: int,
    offset: int,
    default_price_min: float,
    default_price_max: float,
) -> tuple[list[dict], int, float, float]:
    """
    Return (rows, total_count, price_min_used, price_max_used).

    Rows: product_id, product_name, price, average_rating, rating_number.
    Filters by COALESCE(price,0) in [price_min, price_max]; optional ILIKE on product_name.
    """
    pmin = default_price_min if price_min is None else float(price_min)
    pmax = default_price_max if price_max is None else float(price_max)
    if pmin > pmax:
        pmin, pmax = pmax, pmin

    search = (q or "").strip()
    pat = f"%{search}%" if search else None

    where_clauses = [
        "COALESCE(price, 0) >= :pmin",
        "COALESCE(price, 0) <= :pmax",
    ]
    params: dict = {"pmin": pmin, "pmax": pmax}

    if pat is not None:
        where_clauses.append("product_name ILIKE :name_pat")
        params["name_pat"] = pat

    where_sql = " AND ".join(where_clauses)

    count_sql = text(f"SELECT COUNT(*) AS c FROM products WHERE {where_sql}")

    list_sql = text(
        f"""
        SELECT product_id, product_name, price, average_rating, rating_number
        FROM products
        WHERE {where_sql}
        ORDER BY rating_number DESC NULLS LAST, product_id
        LIMIT :lim OFFSET :off
        """
    )

    try:
        with db_engine.connect() as conn:
            total = int(conn.execute(count_sql, params).scalar() or 0)
            params_list = {**params, "lim": limit, "off": offset}
            result = conn.execute(list_sql, params_list)
            rows = []
            for row in result:
                rows.append(
                    {
                        "product_id": str(row[0]),
                        "product_name": str(row[1]) if row[1] is not None else "",
                        "price": float(row[2]) if row[2] is not None else None,
                        "average_rating": float(row[3]) if row[3] is not None else None,
                        "rating_number": float(row[4]) if row[4] is not None else None,
                    }
                )
            return rows, total, pmin, pmax
    except Exception as e:
        logger.error("list_products failed: %s", e, exc_info=True)
        raise
