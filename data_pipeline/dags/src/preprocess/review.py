"""Deterministic preprocessing pipeline for review dataset."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import pandas as pd

from .utils import ensure_output_dir, get_processed_path, get_raw_path, setup_logging


LOGGER = logging.getLogger("preprocess.review")

INPUT_FILENAME = "review_data.jsonl"
OUTPUT_FILENAME = "review_preprocessed.jsonl"
BATCH_SIZE = 50_000

INPUT_COLUMNS: List[str] = [
    "user_id",
    "asin",
    "parent_asin",
    "rating",
    "title",
    "text",
    "verified_purchase",
    "helpful_vote",
]

FINAL_COLUMNS: List[str] = [
    "user_id",
    "asin",
    "product_id",
    "rating",
    "review_title",
    "review_text",
    "verified_purchase",
    "helpful_vote",
]


@dataclass
class PreprocessStats:
    """Counters for deterministic, auditable preprocessing."""

    raw_rows_loaded: int = 0
    malformed_json_rows_skipped: int = 0
    valid_json_rows: int = 0
    removed_missing_parent_asin: int = 0
    removed_missing_user_id: int = 0
    removed_missing_rating: int = 0
    duplicates_removed: int = 0
    final_rows: int = 0


def _normalize_text(value: object) -> str:
    """Convert a value into a non-null text string for embedding safety."""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, str):
        return value.strip()
    if pd.isna(value):
        return ""
    return str(value)


def _normalize_id(value: object) -> str:
    """Normalize ID-like fields to deterministic stripped strings."""
    text = _normalize_text(value)
    if text.lower() == "nan":
        return ""
    return text


def _to_bool(value: object) -> bool:
    """Parse verified_purchase into a strict boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, dict)):
        return False
    if pd.isna(value):
        return False

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "t"}:
        return True
    if normalized in {"false", "0", "no", "n", "f", ""}:
        return False
    return False


def _process_batch(
    records: List[Dict[str, object]],
    seen_review_keys: Set[Tuple[str, str]],
    stats: PreprocessStats,
) -> pd.DataFrame:
    """Apply deterministic transformations to one DataFrame batch."""
    if not records:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    df = pd.DataFrame.from_records(records).reindex(columns=INPUT_COLUMNS)

    # Drop rows where parent_asin is missing/empty.
    parent_asin_norm = df["parent_asin"].apply(_normalize_id)
    missing_parent_mask = parent_asin_norm.eq("")
    removed_parent = int(missing_parent_mask.sum())
    if removed_parent:
        stats.removed_missing_parent_asin += removed_parent
        df = df.loc[~missing_parent_mask].copy()
        parent_asin_norm = parent_asin_norm.loc[~missing_parent_mask]
    df["parent_asin"] = parent_asin_norm

    # Drop rows where user_id is missing/empty.
    user_id_norm = df["user_id"].apply(_normalize_id)
    missing_user_mask = user_id_norm.eq("")
    removed_user = int(missing_user_mask.sum())
    if removed_user:
        stats.removed_missing_user_id += removed_user
        df = df.loc[~missing_user_mask].copy()
        user_id_norm = user_id_norm.loc[~missing_user_mask]
    df["user_id"] = user_id_norm

    # Enforce numeric rating and drop missing ratings.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype(float)
    missing_rating_mask = df["rating"].isna()
    removed_rating = int(missing_rating_mask.sum())
    if removed_rating:
        stats.removed_missing_rating += removed_rating
        df = df.loc[~missing_rating_mask].copy()

    # Embedding-safe text fields (must never be null).
    df["title"] = df["title"].apply(_normalize_text)
    df["text"] = df["text"].apply(_normalize_text)

    # Deterministic type conversions.
    df["asin"] = df["asin"].apply(_normalize_id)
    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype(int)
    df["verified_purchase"] = df["verified_purchase"].apply(_to_bool).astype(bool)

    # Global duplicate key: (asin + user_id)
    keys = list(zip(df["asin"], df["user_id"]))
    keep_mask = []
    for key in keys:
        if key in seen_review_keys:
            keep_mask.append(False)
        else:
            seen_review_keys.add(key)
            keep_mask.append(True)

    duplicate_count = len(keep_mask) - int(sum(keep_mask))
    if duplicate_count:
        stats.duplicates_removed += duplicate_count
        df = df.loc[keep_mask].copy()

    # Rename columns for final output
    df = df.rename(columns={
        "parent_asin": "product_id",
        "title": "review_title",
        "text": "review_text"
    })

    # Final exact schema.
    return df.reindex(columns=FINAL_COLUMNS)


def _print_snapshot(df: pd.DataFrame, title: str, rows: int = 5) -> None:
    """Print compact DataFrame snapshots for observability."""
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    if not df.empty:
        print(df.head(rows).to_string(index=False))


def preprocess_review_data(input_path: str, output_path: str) -> pd.DataFrame:
    """Stream, preprocess, and persist review data as JSONL."""
    ensure_output_dir(output_path)

    stats = PreprocessStats()
    seen_review_keys: Set[Tuple[str, str]] = set()
    discovered_columns: Set[str] = set()
    batch_records: List[Dict[str, object]] = []
    sample_frames: List[pd.DataFrame] = []

    # Write to temp file to preserve existing output for incremental merge.
    new_output_path = output_path + ".new.tmp"
    open(new_output_path, "w", encoding="utf-8").close()

    LOGGER.info("Loading JSONL stream from: %s", input_path)
    with open(input_path, "r", encoding="utf-8") as infile, open(new_output_path, "a", encoding="utf-8") as outfile:
        for raw_line in infile:
            stats.raw_rows_loaded += 1
            line = raw_line.strip()
            if not line:
                stats.malformed_json_rows_skipped += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.malformed_json_rows_skipped += 1
                continue

            if not isinstance(record, dict):
                stats.malformed_json_rows_skipped += 1
                continue

            stats.valid_json_rows += 1
            discovered_columns.update(record.keys())
            batch_records.append(record)

            if len(batch_records) >= BATCH_SIZE:
                cleaned_batch = _process_batch(batch_records, seen_review_keys, stats)
                if not cleaned_batch.empty:
                    for item in cleaned_batch.to_dict(orient="records"):
                        outfile.write(json.dumps(item, ensure_ascii=True) + "\n")
                    stats.final_rows += len(cleaned_batch)

                    # Keep a small sample for printing at the end.
                    if len(sample_frames) < 2:
                        sample_frames.append(cleaned_batch.head(3))
                batch_records.clear()

        # Flush final incomplete batch.
        if batch_records:
            cleaned_batch = _process_batch(batch_records, seen_review_keys, stats)
            if not cleaned_batch.empty:
                for item in cleaned_batch.to_dict(orient="records"):
                    outfile.write(json.dumps(item, ensure_ascii=True) + "\n")
                stats.final_rows += len(cleaned_batch)
                if len(sample_frames) < 2:
                    sample_frames.append(cleaned_batch.head(3))

    print("\nInitial Review Dataset")
    print("----------------------")
    print(f"Initial valid row count: {stats.valid_json_rows}")
    print(f"Initial columns discovered: {sorted(discovered_columns)}")

    # Build a small in-memory sample for preview only.
    if sample_frames:
        final_sample = pd.concat(sample_frames, ignore_index=True).head(5)
    else:
        final_sample = pd.DataFrame(columns=FINAL_COLUMNS)

    _print_snapshot(final_sample, title=f"Final Review Sample (rows saved: {stats.final_rows})", rows=5)

    # --- Incremental merge: merge new output with existing preprocessed file ---
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        from src.incremental import merge_jsonl
        merge_stats = merge_jsonl(
            new_output_path, output_path, key_cols=["user_id", "product_id"]
        )
        os.remove(new_output_path)
        LOGGER.info("Incremental merge stats: %s", merge_stats)
    else:
        # First run — just move temp to output.
        os.replace(new_output_path, output_path)

    LOGGER.info(
        "Dropped fields timestamp/images: not useful for text embeddings or sentiment analytics and add noise."
    )
    LOGGER.info("Raw rows loaded: %d", stats.raw_rows_loaded)
    LOGGER.info("Malformed JSON rows skipped: %d", stats.malformed_json_rows_skipped)
    LOGGER.info("Duplicates removed ((asin + user_id)): %d", stats.duplicates_removed)
    LOGGER.info(
        "Rows dropped (missing IDs or ratings): %d",
        stats.removed_missing_parent_asin + stats.removed_missing_user_id + stats.removed_missing_rating,
    )
    LOGGER.info("Rows dropped (missing parent_asin): %d", stats.removed_missing_parent_asin)
    LOGGER.info("Rows dropped (missing user_id): %d", stats.removed_missing_user_id)
    LOGGER.info("Rows dropped (missing rating): %d", stats.removed_missing_rating)
    LOGGER.info("Final row count: %d", stats.final_rows)
    LOGGER.info("Saved cleaned review data to: %s", output_path)

    return final_sample


def main() -> None:
    """Entry point for deterministic review preprocessing."""
    setup_logging()
    input_path = get_raw_path(INPUT_FILENAME)
    output_path = get_processed_path(OUTPUT_FILENAME, base_dir="data/processed")
    preprocess_review_data(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
