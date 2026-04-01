"""Deterministic streaming preprocessing for product JSONL dataset."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from .utils import ensure_output_dir, get_processed_path, get_raw_path, setup_logging


LOGGER = logging.getLogger("preprocess.product")

INPUT_FILENAME = "product_data.jsonl"
OUTPUT_FILENAME = "product_preprocessed.jsonl"
BATCH_SIZE = 25_000

# Keep-list from requirements.
WORKING_COLUMNS: List[str] = [
    "parent_asin",
    "title",
    "price",
    "average_rating",
    "rating_number",
    "description",
    "features",
    "details",
    "categories",
]

# Final schema requested in step 6.
FINAL_COLUMNS: List[str] = [
    "product_id",
    "product_name",
    "price",
    "average_rating",
    "rating_number",
    "description",
    "features",
    "details",
    "category",
]

TEXT_COLUMNS: List[str] = ["description", "features", "details"]


@dataclass
class ProductStats:
    """Counters for auditable preprocessing logs."""

    raw_rows_loaded: int = 0
    malformed_rows_skipped: int = 0
    valid_rows: int = 0
    removed_missing_parent_asin: int = 0
    duplicates_removed: int = 0
    negative_price_removed: int = 0
    low_price_removed: int = 0
    prices_imputed: int = 0
    price_imputed_title: int = 0
    price_imputed_category: int = 0
    price_fallback_count: int = 0
    final_row_count: int = 0


def _print_frame_snapshot(df: pd.DataFrame, title: str, rows: int = 5) -> None:
    """Print compact frame preview."""
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    if not df.empty:
        print(df.head(rows).to_string(index=False))


def _normalize_text(value: object) -> str:
    """Convert any field to deterministic non-null string."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _normalize_categories(value: object) -> str:
    """Normalize categories: list -> ' > ', missing -> ''."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, list):
        parts = [_normalize_text(item) for item in value]
        return " > ".join([p for p in parts if p])
    return _normalize_text(value)


def _normalize_parent_asin(value: object) -> str:
    """Normalize parent_asin and collapse null-like tokens."""
    text = _normalize_text(value)
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _title_group_key(title: object, token_limit: int = 3) -> str:
    """Create deterministic title keyword key used for local pricing groups."""
    title_text = _normalize_text(title).lower()
    tokens = re.findall(r"[a-z0-9]+", title_text)
    return "_".join(tokens[:token_limit]) if tokens else "__unknown__"


def _process_stage1_batch(
    records: List[Dict[str, object]],
    seen_parent_asins: Set[str],
    stats: ProductStats,
) -> pd.DataFrame:
    """Apply selection, id validation, dedupe, and base type cleaning."""
    if not records:
        return pd.DataFrame(columns=WORKING_COLUMNS)

    df = pd.DataFrame.from_records(records).reindex(columns=WORKING_COLUMNS)

    # Identifier validation.
    df["parent_asin"] = df["parent_asin"].apply(_normalize_parent_asin)
    missing_parent_mask = df["parent_asin"].eq("")
    removed_missing_parent = int(missing_parent_mask.sum())
    if removed_missing_parent:
        stats.removed_missing_parent_asin += removed_missing_parent
        df = df.loc[~missing_parent_mask].copy()

    # Global dedupe on parent_asin.
    keep_mask = []
    for key in df["parent_asin"]:
        if key in seen_parent_asins:
            keep_mask.append(False)
        else:
            seen_parent_asins.add(key)
            keep_mask.append(True)
    duplicates_removed = len(keep_mask) - int(sum(keep_mask))
    if duplicates_removed:
        stats.duplicates_removed += duplicates_removed
        df = df.loc[keep_mask].copy()

    # Base type conversions.
    df["title"] = df["title"].apply(_normalize_text)
    df["description"] = df["description"].apply(_normalize_text)
    df["features"] = df["features"].apply(_normalize_text)
    df["details"] = df["details"].apply(_normalize_text)
    df["categories"] = df["categories"].apply(_normalize_categories)

    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float)
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce").astype(float)
    df["rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce").fillna(0).astype(int)

    # Step A price validation for non-missing prices only.
    negative_price_mask = df["price"].notna() & (df["price"] < 0)
    negative_price_removed = int(negative_price_mask.sum())
    if negative_price_removed:
        stats.negative_price_removed += negative_price_removed
        df = df.loc[~negative_price_mask].copy()

    low_price_mask = df["price"].notna() & (df["price"] == 0)
    low_price_removed = int(low_price_mask.sum())
    if low_price_removed:
        stats.low_price_removed += low_price_removed
        df = df.loc[~low_price_mask].copy()

    return df.reindex(columns=WORKING_COLUMNS)


def preprocess_product_data(input_path: str, output_path: str) -> pd.DataFrame:
    """Run streaming deterministic product preprocessing."""
    ensure_output_dir(output_path)

    stage1_path = f"{output_path}.stage1.tmp"
    new_output_path = f"{output_path}.new.tmp"  # write new data here to preserve existing
    # Reset stage files for deterministic reruns.
    open(stage1_path, "w", encoding="utf-8").close()
    open(new_output_path, "w", encoding="utf-8").close()

    stats = ProductStats()
    seen_parent_asins: Set[str] = set()
    discovered_columns: Set[str] = set()
    batch_records: List[Dict[str, object]] = []
    sample_frames: List[pd.DataFrame] = []

    # For imputation statistics (built from valid, non-missing prices only).
    group_price_values: Dict[str, List[float]] = {}
    category_price_values: Dict[str, List[float]] = {}
    all_valid_prices: List[float] = []

    LOGGER.info("Loading JSONL stream from: %s", input_path)
    with open(input_path, "r", encoding="utf-8") as infile, open(stage1_path, "a", encoding="utf-8") as stage1:
        for raw_line in infile:
            stats.raw_rows_loaded += 1
            line = raw_line.strip()
            if not line:
                stats.malformed_rows_skipped += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.malformed_rows_skipped += 1
                continue

            if not isinstance(record, dict):
                stats.malformed_rows_skipped += 1
                continue

            stats.valid_rows += 1
            discovered_columns.update(record.keys())
            batch_records.append(record)

            if len(batch_records) >= BATCH_SIZE:
                cleaned = _process_stage1_batch(batch_records, seen_parent_asins, stats)
                if not cleaned.empty:
                    cleaned["title_group_key"] = cleaned["title"].apply(_title_group_key)
                    cleaned["category_key"] = cleaned["categories"].apply(_normalize_categories)
                    priced = cleaned[cleaned["price"].notna()]
                    for _, row in priced[["title_group_key", "category_key", "price"]].iterrows():
                        group_key = str(row["title_group_key"])
                        cat_key = str(row["category_key"])
                        price = float(row["price"])
                        group_price_values.setdefault(group_key, []).append(price)
                        if cat_key:
                            category_price_values.setdefault(cat_key, []).append(price)
                        all_valid_prices.append(price)

                    for item in cleaned.to_dict(orient="records"):
                        stage1.write(json.dumps(item, ensure_ascii=True) + "\n")

                    if len(sample_frames) < 2:
                        sample_frames.append(cleaned.head(3))
                batch_records.clear()

        if batch_records:
            cleaned = _process_stage1_batch(batch_records, seen_parent_asins, stats)
            if not cleaned.empty:
                cleaned["title_group_key"] = cleaned["title"].apply(_title_group_key)
                cleaned["category_key"] = cleaned["categories"].apply(_normalize_categories)
                priced = cleaned[cleaned["price"].notna()]
                for _, row in priced[["title_group_key", "category_key", "price"]].iterrows():
                    group_key = str(row["title_group_key"])
                    cat_key = str(row["category_key"])
                    price = float(row["price"])
                    group_price_values.setdefault(group_key, []).append(price)
                    if cat_key:
                        category_price_values.setdefault(cat_key, []).append(price)
                    all_valid_prices.append(price)

                for item in cleaned.to_dict(orient="records"):
                    stage1.write(json.dumps(item, ensure_ascii=True) + "\n")
                if len(sample_frames) < 2:
                    sample_frames.append(cleaned.head(3))

    if not all_valid_prices:
        raise ValueError("No valid non-missing prices remain after Step A; cannot impute missing prices.")

    global_median = float(np.median(all_valid_prices))
    group_medians: Dict[str, float] = {}
    for key, prices in group_price_values.items():
        arr = np.asarray(prices, dtype=float)
        group_medians[key] = float(np.median(arr))

    category_medians: Dict[str, float] = {}
    for key, prices in category_price_values.items():
        arr = np.asarray(prices, dtype=float)
        category_medians[key] = float(np.median(arr))

    LOGGER.info(
        "Dropped non-required fields (main_category, store, images, videos) for cleaner analytics/embeddings."
    )
    LOGGER.info(
        "Built price stats for %d title groups and %d categories (global median: $%.2f).",
        len(group_medians), len(category_medians), global_median,
    )

    # Pass 2: impute missing prices and write final output.
    final_samples: List[Dict[str, object]] = []
    with open(stage1_path, "r", encoding="utf-8") as stage1, open(new_output_path, "a", encoding="utf-8") as out:
        for raw_line in stage1:
            record = json.loads(raw_line)
            price_val = record.get("price")
            if price_val is None or (isinstance(price_val, float) and np.isnan(price_val)):
                group_key = _title_group_key(record.get("title", ""))
                cat_key = _normalize_categories(record.get("categories"))
                if group_key in group_medians:
                    record["price"] = float(group_medians[group_key])
                    stats.price_imputed_title += 1
                elif cat_key and cat_key in category_medians:
                    record["price"] = float(category_medians[cat_key])
                    stats.price_imputed_category += 1
                else:
                    record["price"] = float(global_median)
                    stats.price_fallback_count += 1
                stats.prices_imputed += 1

            # RENAME FIELDS FOR FINAL OUPUT
            final_record = {
                "product_id": record.get("parent_asin"),
                "product_name": record.get("title"),
                "price": record.get("price"),
                "average_rating": record.get("average_rating"),
                "rating_number": record.get("rating_number"),
                "description": record.get("description"),
                "features": record.get("features"),
                "details": record.get("details"),
                "category": record.get("categories"),
            }
            out.write(json.dumps(final_record, ensure_ascii=True) + "\n")
            stats.final_row_count += 1
            if len(final_samples) < 5:
                final_samples.append(final_record)

    os.remove(stage1_path)

    # --- Incremental merge: merge new output with existing preprocessed file ---
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        from src.incremental import merge_jsonl
        merge_stats = merge_jsonl(new_output_path, output_path, key_cols=["product_id"])
        os.remove(new_output_path)
        LOGGER.info("Incremental merge stats: %s", merge_stats)
    else:
        # First run — just move temp to output.
        os.replace(new_output_path, output_path)

    initial_view = pd.DataFrame(columns=sorted(discovered_columns))
    _print_frame_snapshot(
        initial_view,
        title=f"Initial Product Dataset (valid rows: {stats.valid_rows})",
        rows=0,
    )

    final_sample_df = pd.DataFrame(final_samples, columns=FINAL_COLUMNS)
    _print_frame_snapshot(
        final_sample_df,
        title=f"Final Product Dataset (rows saved: {stats.final_row_count})",
        rows=5,
    )

    LOGGER.info("Raw rows loaded: %d", stats.raw_rows_loaded)
    LOGGER.info("Malformed JSON rows skipped: %d", stats.malformed_rows_skipped)
    LOGGER.info("Rows removed (missing/empty parent_asin): %d", stats.removed_missing_parent_asin)
    LOGGER.info("Duplicates removed (by parent_asin): %d", stats.duplicates_removed)
    LOGGER.info("Rows removed (price < 0): %d", stats.negative_price_removed)
    LOGGER.info("Rows removed (price == 0): %d", stats.low_price_removed)
    LOGGER.info("Prices imputed (missing -> estimated): %d", stats.prices_imputed)
    LOGGER.info("  ├─ via title-group median: %d", stats.price_imputed_title)
    LOGGER.info("  ├─ via category median: %d", stats.price_imputed_category)
    LOGGER.info("  └─ via global median (fallback): %d", stats.price_fallback_count)
    LOGGER.info("Final row count: %d", stats.final_row_count)
    LOGGER.info("Saved cleaned product dataset to: %s", output_path)

    return final_sample_df


def main() -> None:
    """Entry point for streaming product preprocessing."""
    setup_logging()
    input_path = get_raw_path(INPUT_FILENAME)
    output_path = get_processed_path(OUTPUT_FILENAME, base_dir="data/processed")
    preprocess_product_data(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
