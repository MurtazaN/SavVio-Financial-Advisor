"""
Bias Detection for SavVio Product Data.

Analyzes all columns in product_preprocessed.jsonl + product_featured.jsonl
for representation gaps, underrepresented groups, and missingness bias.

Input: data/processed/product_preprocessed.jsonl
       data/features/product_featured.jsonl (optional merge)
Output: Terminal-only log output (no files written)
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

BOOL_TRUE = {"1", "true", "yes", "y", "t"}
BOOL_FALSE = {"0", "false", "no", "n", "f"}
MISSING_STRINGS = {"", "na", "n/a", "nan", "null", "none", "missing"}
COMMON_DETAIL_KEYS = ["Brand", "Model", "Capacity", "Date First Available"]


@dataclass
class SliceStat:
    label: str
    count: int
    pct: float


@dataclass
class FlagItem:
    column: str
    group: str
    reason: str


def _norm(text: str) -> str:
    return text.strip().lower()


def _pct(count: int, total: int) -> float:
    return round((count / total * 100.0), 2) if total else 0.0


def _safe_json_parse(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if (stripped.startswith("[") and stripped.endswith("]")) or (
            stripped.startswith("{") and stripped.endswith("}")
        ):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return None
    return None


def _first_existing(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lookup = {_norm(c): c for c in df.columns}
    for name in names:
        if _norm(name) in lookup:
            return lookup[_norm(name)]
    return None


def _missing_mask_generic(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.isna()
    lowered = series.astype(str).str.strip().str.lower()
    return series.isna() | lowered.isin(MISSING_STRINGS)


def _is_list_column(name: str, series: pd.Series) -> bool:
    if _norm(name) in {"features", "description", "categories", "images", "videos"}:
        return True
    parsed = series.dropna().head(200).apply(_safe_json_parse)
    return bool(parsed.apply(lambda x: isinstance(x, list)).mean() > 0.7)


def _is_dict_column(name: str, series: pd.Series) -> bool:
    if _norm(name) == "details":
        return True
    parsed = series.dropna().head(200).apply(_safe_json_parse)
    return bool(parsed.apply(lambda x: isinstance(x, dict)).mean() > 0.7)


# ---------------------------------------------------------------------------
# Column type inference
# ---------------------------------------------------------------------------

def _infer_column_type(name: str, series: pd.Series) -> str:
    n = _norm(name)
    non_missing = series[~_missing_mask_generic(series)]
    if non_missing.empty:
        return "categorical"

    if n in {"product_id", "parent_asin"}:
        uniqueness = non_missing.nunique(dropna=True) / max(len(non_missing), 1)
        return "id" if uniqueness >= 0.95 else "categorical"

    if _is_list_column(name, series):
        return "list"
    if _is_dict_column(name, series):
        return "dict"

    lowered = non_missing.astype(str).str.strip().str.lower()
    if lowered.isin(BOOL_TRUE | BOOL_FALSE).all() or pd.api.types.is_bool_dtype(series):
        return "boolean"

    num = pd.to_numeric(non_missing, errors="coerce")
    if pd.api.types.is_numeric_dtype(series) or num.notna().mean() >= 0.95:
        return "numeric"

    if n in {"title", "product_name"}:
        return "text"
    if non_missing.astype(str).str.len().mean() > 80:
        return "text"
    return "categorical"


# ---------------------------------------------------------------------------
# Slice helpers
# ---------------------------------------------------------------------------

def _slice_counts(labels: pd.Series, total: int, sort_index: bool = False) -> List[SliceStat]:
    counts = labels.value_counts(dropna=False, sort=not sort_index)
    if sort_index:
        counts = counts.sort_index()
    return [SliceStat(str(k), int(v), _pct(int(v), total)) for k, v in counts.items()]


def _log_slice_stats(stats: List[SliceStat]) -> None:
    for s in stats:
        logger.info(f"  - {s.label}: {s.count} ({s.pct}%)")


# ---------------------------------------------------------------------------
# Numeric banding (domain-specific)
# ---------------------------------------------------------------------------

def _numeric_slices(column: str, numeric: pd.Series) -> Tuple[pd.Series, List[str]]:
    name = _norm(column)
    labels = pd.Series(index=numeric.index, dtype="object")
    risk_targets: List[str] = []

    if name == "price":
        labels[numeric.isna()] = "Missing"
        labels[numeric < 25] = "Budget"
        labels[(numeric >= 25) & (numeric <= 200)] = "Mid-range"
        labels[numeric > 200] = "Premium"
        risk_targets = ["Missing", "Budget"]
        return labels, risk_targets

    if name == "average_rating":
        labels[numeric.isna()] = "Missing"
        labels[numeric <= 3.0] = "Low"
        labels[(numeric > 3.0) & (numeric <= 4.0)] = "Medium"
        labels[numeric > 4.0] = "High"
        risk_targets = ["Low"]
        return labels, risk_targets

    if name in {"rating_number", "num_reviews"}:
        labels[numeric.isna()] = "Missing"
        labels[numeric < 10] = "Low-confidence"
        labels[(numeric >= 10) & (numeric <= 100)] = "Medium-confidence"
        labels[numeric > 100] = "High-confidence"
        risk_targets = ["Low-confidence"]
        return labels, risk_targets

    if name == "rating_variance":
        labels[numeric.isna()] = "Missing"
        labels[numeric == 0.0] = "Single-review proxy"
        labels[(numeric > 0.0) & (numeric < 0.5)] = "Consensus"
        labels[(numeric >= 0.5) & (numeric <= 1.0)] = "Mixed"
        labels[numeric > 1.0] = "Polarized"
        risk_targets = ["Single-review proxy", "Polarized"]
        return labels, risk_targets

    labels[numeric.isna()] = "Missing"
    valid = numeric.dropna()
    if valid.empty:
        return labels, []
    try:
        q = pd.qcut(valid, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        labels.loc[q.index] = q.astype(str)
    except ValueError:
        labels.loc[valid.index] = "Q2"
    labels[valid >= valid.quantile(0.99)] = "Outlier (Top 1%)"
    return labels, []


# ---------------------------------------------------------------------------
# Type-specific profilers
# ---------------------------------------------------------------------------

def _categorical_profile(column: str, series: pd.Series, total: int) -> Tuple[List[SliceStat], List[FlagItem]]:
    labels = series.astype(str).str.strip().where(~_missing_mask_generic(series), "Missing")
    stats = _slice_counts(labels, total)
    flags: List[FlagItem] = []

    for row in stats:
        if row.label != "Missing" and row.pct < 5.0:
            flags.append(FlagItem(column, row.label, f"Category underrepresented (<5%): {row.pct}%"))
        if row.label != "Missing" and row.pct > 70.0:
            flags.append(FlagItem(column, row.label, f"Category domination risk (>70%): {row.pct}%"))
    return stats, flags


def _list_profile(column: str, series: pd.Series, total: int) -> Tuple[List[SliceStat], List[FlagItem]]:
    parsed = series.apply(_safe_json_parse)
    lengths = pd.Series(index=series.index, dtype="float")

    for idx, value in parsed.items():
        if isinstance(value, list):
            lengths.at[idx] = len(value)
        elif _missing_mask_generic(series.loc[[idx]]).iloc[0]:
            lengths.at[idx] = 0
        else:
            lengths.at[idx] = math.nan

    labels = pd.Series(index=series.index, dtype="object")
    labels[lengths.isna()] = "Invalid"
    labels[lengths == 0] = "0"
    labels[(lengths >= 1) & (lengths <= 2)] = "1-2"
    labels[(lengths >= 3) & (lengths <= 5)] = "3-5"
    labels[lengths >= 6] = "6+"

    stats = _slice_counts(labels, total)
    flags: List[FlagItem] = []
    empty_share = _pct(int((labels == "0").sum()), total)

    n = _norm(column)
    if n in {"features", "categories"} and empty_share > 20.0:
        flags.append(FlagItem(column, "0", f"Empty list share >20% ({empty_share}%)."))
    if n == "images" and empty_share > 10.0:
        flags.append(FlagItem(column, "0", f"Images empty share >10% ({empty_share}%)."))
    return stats, flags


def _details_profile(column: str, series: pd.Series, total: int) -> Tuple[List[SliceStat], List[FlagItem]]:
    parsed = series.apply(_safe_json_parse)
    has_dict = parsed.apply(lambda x: isinstance(x, dict))
    empty_or_missing = (~has_dict) | parsed.apply(lambda x: isinstance(x, dict) and len(x) == 0)

    labels = pd.Series(index=series.index, dtype="object")
    labels[empty_or_missing] = "Missing/Empty"
    labels[~empty_or_missing] = "Present"
    stats = _slice_counts(labels, total)

    flags: List[FlagItem] = []
    empty_share = _pct(int(empty_or_missing.sum()), total)
    if empty_share > 20.0:
        flags.append(FlagItem(column, "Missing/Empty", f"Details missing/empty >20% ({empty_share}%)."))

    # Key-level completeness + brand distribution.
    key_rows: Dict[str, int] = {k: 0 for k in COMMON_DETAIL_KEYS}
    brand_values: List[str] = []
    for value in parsed:
        if isinstance(value, dict):
            for key in COMMON_DETAIL_KEYS:
                if key in value and str(value[key]).strip() != "":
                    key_rows[key] += 1
            if "Brand" in value and str(value["Brand"]).strip() != "":
                brand_values.append(str(value["Brand"]).strip())

    logger.info("  - details key missingness:")
    for key in COMMON_DETAIL_KEYS:
        miss_pct = _pct(total - key_rows[key], total)
        logger.info(f"    - {key}: {miss_pct}% missing")

    if brand_values:
        brand_counts = pd.Series(brand_values).value_counts(normalize=True) * 100.0
        logger.info("  - Brand distribution (top 10):")
        for brand, pct in brand_counts.head(10).items():
            logger.info(f"    - {brand}: {round(float(pct), 2)}%")
        rare_brands = brand_counts[brand_counts < 5.0]
        if not rare_brands.empty:
            sample_rare = ", ".join([f"{b} ({round(float(p), 2)}%)" for b, p in rare_brands.head(5).items()])
            flags.append(FlagItem(column, "Brand", f"Rare brands (<5%) present: {sample_rare}"))

    return stats, flags


def _text_profile(column: str, series: pd.Series, total: int) -> Tuple[List[SliceStat], List[FlagItem]]:
    txt = series.astype(str).where(~_missing_mask_generic(series), "")
    lengths = txt.str.len()
    labels = pd.Series(index=series.index, dtype="object")
    labels[lengths == 0] = "Missing/Empty"
    labels[(lengths > 0) & (lengths < 40)] = "Short"
    labels[(lengths >= 40) & (lengths <= 120)] = "Medium"
    labels[lengths > 120] = "Long"
    stats = _slice_counts(labels, total)

    flags: List[FlagItem] = []
    missing_share = _pct(int((labels == "Missing/Empty").sum()), total)
    if _norm(column) in {"title", "product_name"} and missing_share > 5.0:
        flags.append(FlagItem(column, "Missing/Empty", f"Title missing/empty >5% ({missing_share}%)."))
    return stats, flags


# ---------------------------------------------------------------------------
# Column profiler dispatcher
# ---------------------------------------------------------------------------

def _profile_column(column: str, series: pd.Series, total: int) -> List[FlagItem]:
    inferred = _infer_column_type(column, series)
    missing_pct = _pct(int(_missing_mask_generic(series).sum()), total)
    flags: List[FlagItem] = []

    logger.info(f"\nColumn: {column}")
    logger.info(f"- inferred type: {inferred}")
    logger.info(f"- missing rate (%%): {missing_pct}")
    logger.info("- representation slices:")

    if missing_pct > 20.0:
        flags.append(FlagItem(column, "Missingness", f"Column missingness >20% ({missing_pct}%)."))
    if _norm(column) == "price" and missing_pct > 5.0:
        flags.append(FlagItem(column, "Missingness", f"Price missingness is high ({missing_pct}%)."))

    if inferred == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        labels, risk_targets = _numeric_slices(column, numeric)
        stats = _slice_counts(labels, total)
        _log_slice_stats(stats)
        for target in risk_targets:
            share = _pct(int((labels == target).sum()), total)
            if share < 5.0:
                flags.append(FlagItem(column, target, f"Uncertainty/negative slice <5% or missing ({share}%)."))
    elif inferred == "categorical":
        stats, f = _categorical_profile(column, series, total)
        _log_slice_stats(stats)
        flags.extend(f)
    elif inferred == "boolean":
        lowered = series.astype(str).str.strip().str.lower()
        labels = pd.Series(index=series.index, dtype="object")
        labels[lowered.isin(BOOL_TRUE)] = "True"
        labels[lowered.isin(BOOL_FALSE)] = "False"
        labels[_missing_mask_generic(series)] = "Missing"
        labels[labels.isna()] = "Invalid"
        stats = _slice_counts(labels, total)
        _log_slice_stats(stats)
    elif inferred == "text":
        stats, f = _text_profile(column, series, total)
        _log_slice_stats(stats)
        flags.extend(f)
    elif inferred == "list":
        stats, f = _list_profile(column, series, total)
        _log_slice_stats(stats)
        flags.extend(f)
    elif inferred == "dict":
        stats, f = _details_profile(column, series, total)
        _log_slice_stats(stats)
        flags.extend(f)
    else:  # id
        non_missing = series[~_missing_mask_generic(series)]
        uniq_rate = _pct(int(non_missing.nunique(dropna=True)), max(len(non_missing), 1))
        _log_slice_stats([SliceStat("Unique non-missing rate", int(non_missing.nunique(dropna=True)), uniq_rate)])
        if uniq_rate < 95.0:
            flags.append(FlagItem(column, "Uniqueness", f"Identifier uniqueness below 95% ({uniq_rate}%)."))

    if flags:
        logger.info("- flagged risks:")
        for f in flags:
            logger.info(f"  - {f.group}: {f.reason}")
    return flags


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_products(preprocessed_path: str, featured_path: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"Preprocessed file not found: {preprocessed_path}")
    df = pd.read_json(preprocessed_path, lines=True)
    logger.info(f"Loaded {len(df)} records from {preprocessed_path}")

    if featured_path and os.path.exists(featured_path):
        feat = pd.read_json(featured_path, lines=True)
        left_key = _first_existing(df, ["product_id", "parent_asin"])
        right_key = _first_existing(feat, ["product_id", "parent_asin"])
        if left_key and right_key:
            dedup_feat = feat.drop(
                columns=[c for c in feat.columns if c in df.columns and c != right_key],
                errors="ignore",
            )
            df = df.merge(dedup_feat, left_on=left_key, right_on=right_key, how="left")
            if left_key != right_key and right_key in df.columns:
                df = df.drop(columns=[right_key])
        logger.info(f"Merged featured data from {featured_path} ({len(df.columns)} total columns)")
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_product_bias(preprocessed_path: str, featured_path: Optional[str] = None) -> None:
    """
    Executes the product bias detection pipeline.
    """
    logger.info("Starting product bias detection...")
    df = _load_products(preprocessed_path, featured_path)
    total_rows = len(df)

    logger.info(f"\n1) Dataset Summary")
    logger.info("-" * 40)
    logger.info(f"total records: {total_rows}, total columns: {len(df.columns)}")
    overall_missing = round(df.isna().sum().sum() / max(df.size, 1) * 100.0, 2)
    logger.info(f"overall missingness rate: {overall_missing}%")
    missing_by_col = {c: _pct(int(_missing_mask_generic(df[c]).sum()), total_rows) for c in df.columns}
    top5 = sorted(missing_by_col.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("top 5 columns by missingness:")
    for col, pct in top5:
        logger.info(f"- {col}: {pct}%")

    logger.info(f"\n2) Column-by-column analysis")
    logger.info("-" * 40)
    all_flags: List[FlagItem] = []
    for column in df.columns:
        all_flags.extend(_profile_column(column, df[column], total_rows))

    logger.info(f"\n3) Final Summary")
    logger.info("-" * 40)
    logger.info("Flagged Representation Risks:")
    if all_flags:
        for f in all_flags:
            logger.info(f"- {f.column} | {f.group}: {f.reason}")
    else:
        logger.info("- None")

    logger.info("\nTraining-time-only mitigation recommendations:")
    if all_flags:
        logger.info("- Stratified sampling by (price band x rating confidence band).")
        logger.info("- Controlled oversampling of underrepresented slices (<5%).")
        logger.info("- Down-weight low-confidence products (rating_variance == 0 or rating_number < 10).")
        logger.info("- Stress-test evaluation on budget, low-rated, and polarized-variance products.")
    else:
        logger.info("- No flagged risks; continue standard monitoring and periodic re-checks.")

    logger.info("Product bias detection complete.")


if __name__ == "__main__":
    from utils import setup_logging, get_processed_path, get_features_path

    setup_logging()
    run_product_bias(
        preprocessed_path=get_processed_path("product_preprocessed.jsonl"),
        featured_path=get_features_path("product_featured.jsonl"),
    )
