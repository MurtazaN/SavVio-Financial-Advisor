"""
Bias Detection for SavVio Review Data.

Analyzes all columns in review_preprocessed.jsonl + review_featured.jsonl
for representation gaps, underrepresented groups, and missingness bias.

Input: data/processed/review_preprocessed.jsonl
       data/features/review_featured.jsonl (optional merge)
Output: Terminal-only log output (no files written)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

BOOL_TRUE = {"1", "true", "yes", "y", "t"}
BOOL_FALSE = {"0", "false", "no", "n", "f"}
MISSING_STRINGS = {"", "na", "n/a", "nan", "null", "none", "missing"}


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


def _norm(name: str) -> str:
    return name.strip().lower()


def _pct(count: int, total: int) -> float:
    return round((count / total * 100.0), 2) if total else 0.0


def _missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.isna()
    lowered = series.astype(str).str.strip().str.lower()
    return series.isna() | lowered.isin(MISSING_STRINGS)


def _counts(series: pd.Series, total: int, sort_index: bool = False) -> List[SliceStat]:
    counts = series.value_counts(dropna=False, sort=not sort_index)
    if sort_index:
        counts = counts.sort_index()
    return [SliceStat(str(k), int(v), _pct(int(v), total)) for k, v in counts.items()]


def _log_slice_stats(stats: List[SliceStat], indent: str = "  ") -> None:
    for row in stats:
        logger.info(f"{indent}- {row.label}: {row.count} ({row.pct}%)")


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lookup = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if _norm(cand) in lookup:
            return lookup[_norm(cand)]
    return None


# ---------------------------------------------------------------------------
# Column type inference
# ---------------------------------------------------------------------------

def _infer_type(col: str, series: pd.Series) -> str:
    name = _norm(col)
    if name in {"user_id", "asin", "product_id"}:
        return "id"
    if name == "verified_purchase":
        return "boolean"
    if name in {"review_title", "review_text"}:
        return "text"

    non_missing = series[~_missing_mask(series)]
    if non_missing.empty:
        return "categorical"

    lowered = non_missing.astype(str).str.strip().str.lower()
    if lowered.isin(BOOL_TRUE | BOOL_FALSE).all() or pd.api.types.is_bool_dtype(series):
        return "boolean"

    numeric = pd.to_numeric(non_missing, errors="coerce")
    if pd.api.types.is_numeric_dtype(series) or numeric.notna().mean() >= 0.95:
        return "numeric"

    uniq_ratio = non_missing.nunique(dropna=True) / max(len(non_missing), 1)
    if uniq_ratio > 0.98:
        return "id"
    if non_missing.astype(str).str.len().mean() > 60:
        return "text"
    return "categorical"


# ---------------------------------------------------------------------------
# Column profilers (domain-specific)
# ---------------------------------------------------------------------------

def _profile_user_id(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    missing_rate = _pct(int(_missing_mask(series).sum()), total)
    non_missing = series[~_missing_mask(series)]
    unique_count = int(non_missing.nunique(dropna=True))
    unique_rate = _pct(unique_count, max(len(non_missing), 1))

    logger.info(f"column name: {col}")
    logger.info("inferred type: id")
    logger.info(f"missing rate (%%): {missing_rate}")
    logger.info("representation slices (bands):")
    _log_slice_stats([SliceStat("uniqueness rate (non-missing)", unique_count, unique_rate)])

    flags: List[FlagItem] = []
    if missing_rate > 0:
        flags.append(FlagItem(col, "missing", f"Missing > 0% ({missing_rate}%)."))
    if unique_rate < 95.0:
        flags.append(FlagItem(col, "uniqueness", f"Uniqueness < 95% ({unique_rate}%)."))

    if flags:
        logger.info("flagged representation risks:")
        for flag in flags:
            logger.info(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_product_identifier(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    missing_rate = _pct(int(_missing_mask(series).sum()), total)
    valid = series[~_missing_mask(series)].astype(str)
    unique_products = int(valid.nunique(dropna=True))
    review_counts = valid.value_counts()

    band_labels = pd.Series(index=review_counts.index, dtype="object")
    band_labels[review_counts == 1] = "1 review"
    band_labels[(review_counts >= 2) & (review_counts <= 5)] = "2-5 reviews"
    band_labels[(review_counts >= 6) & (review_counts <= 20)] = "6-20 reviews"
    band_labels[review_counts >= 21] = "21+ reviews"
    band_stats = _counts(band_labels, max(unique_products, 1))

    logger.info(f"column name: {col}")
    logger.info("inferred type: id")
    logger.info(f"missing rate (%%): {missing_rate}")
    logger.info("representation slices (bands):")
    logger.info(f"  - unique products: {unique_products}")
    _log_slice_stats(band_stats)

    flags: List[FlagItem] = []
    single_review_share = _pct(int((band_labels == "1 review").sum()), max(unique_products, 1))
    if single_review_share > 60.0:
        flags.append(
            FlagItem(col, "1 review", f"Single-review products dominate >60% ({single_review_share}%).")
        )

    if flags:
        logger.info("flagged representation risks:")
        for flag in flags:
            logger.info(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_rating(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    num = pd.to_numeric(series, errors="coerce")
    labels = pd.Series(index=num.index, dtype="object")
    labels[num.isna()] = "Missing"
    labels[(num >= 1) & (num <= 2)] = "Negative"
    labels[num == 3] = "Neutral"
    labels[(num >= 4) & (num <= 5)] = "Positive"
    labels[labels.isna()] = "Invalid"
    stats = _counts(labels, total)

    logger.info(f"column name: {col}")
    logger.info("inferred type: numeric")
    logger.info(f"missing rate (%%): {_pct(int((labels == 'Missing').sum()), total)}")
    logger.info("representation slices (bands):")
    _log_slice_stats(stats)

    flags: List[FlagItem] = []
    neg_share = _pct(int((labels == "Negative").sum()), total)
    neu_share = _pct(int((labels == "Neutral").sum()), total)
    pos_count = int((labels == "Positive").sum())
    if neg_share < 5.0:
        flags.append(FlagItem(col, "Negative", f"Negative <5% or missing ({neg_share}%)."))
    if neu_share < 5.0:
        flags.append(FlagItem(col, "Neutral", f"Neutral <5% or missing ({neu_share}%)."))
    if pos_count == 0 or int((labels == "Negative").sum()) == 0 or int((labels == "Neutral").sum()) == 0:
        flags.append(FlagItem(col, "bucket coverage", "At least one rating bucket has 0%."))

    if flags:
        logger.info("flagged representation risks:")
        for flag in flags:
            logger.info(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_helpful_vote(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    num = pd.to_numeric(series, errors="coerce")
    labels = pd.Series(index=num.index, dtype="object")
    labels[num.isna()] = "Missing"
    labels[num == 0] = "None"
    labels[(num >= 1) & (num <= 2)] = "Low"
    labels[(num >= 3) & (num <= 10)] = "Medium"
    labels[num > 10] = "High"
    labels[labels.isna()] = "Invalid"
    stats = _counts(labels, total)

    logger.info(f"column name: {col}")
    logger.info("inferred type: numeric")
    logger.info(f"missing rate (%%): {_pct(int((labels == 'Missing').sum()), total)}")
    logger.info("representation slices (bands):")
    _log_slice_stats(stats)

    flags: List[FlagItem] = []
    high_share = _pct(int((labels == "High").sum()), total)
    none_share = _pct(int((labels == "None").sum()), total)
    if high_share < 5.0:
        flags.append(FlagItem(col, "High", f"High helpful votes <5% ({high_share}%)."))
    if none_share > 90.0:
        flags.append(FlagItem(
            col, "None",
            f"Low usefulness signal coverage: helpful_vote==0 exceeds 90% ({none_share}%).",
        ))

    if flags:
        logger.info("flagged representation risks:")
        for flag in flags:
            logger.info(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_verified_purchase(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    lowered = series.astype(str).str.strip().str.lower()
    labels = pd.Series(index=series.index, dtype="object")
    labels[lowered.isin(BOOL_TRUE)] = "True"
    labels[lowered.isin(BOOL_FALSE)] = "False"
    labels[_missing_mask(series)] = "Missing"
    labels[labels.isna()] = "Invalid"
    stats = _counts(labels, total)

    logger.info(f"column name: {col}")
    logger.info("inferred type: boolean")
    logger.info(f"missing rate (%%): {_pct(int((labels == 'Missing').sum()), total)}")
    logger.info("representation slices (bands):")
    _log_slice_stats(stats)

    flags: List[FlagItem] = []
    valid = labels[labels.isin(["True", "False"])]
    if not valid.empty:
        dist = valid.value_counts(normalize=True)
        minority = float(dist.min() * 100.0)
        minority_label = str(dist.idxmin())
        if minority < 5.0:
            flags.append(FlagItem(col, minority_label, f"Minority class <5% ({round(minority, 2)}%)."))

    if flags:
        logger.info("flagged representation risks:")
        for flag in flags:
            logger.info(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_review_title(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    txt = series.astype(str).where(~_missing_mask(series), "")
    length = txt.str.len()
    labels = pd.Series(index=series.index, dtype="object")
    labels[length == 0] = "Empty"
    labels[(length > 0) & (length < 15)] = "Short"
    labels[(length >= 15) & (length <= 60)] = "Medium"
    labels[length > 60] = "Long"
    stats = _counts(labels, total)

    missing_share = _pct(int((labels == "Empty").sum()), total)
    logger.info(f"column name: {col}")
    logger.info("inferred type: text")
    logger.info(f"missing rate (%%): {missing_share}")
    logger.info("representation slices (bands):")
    _log_slice_stats(stats)

    flags: List[FlagItem] = []
    if missing_share > 20.0:
        flags.append(FlagItem(col, "Empty", f"Empty/missing >20% ({missing_share}%)."))
    if flags:
        logger.info("flagged representation risks:")
        for flag in flags:
            logger.info(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_review_text(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    txt = series.astype(str).where(~_missing_mask(series), "")
    length = txt.str.len()
    labels = pd.Series(index=series.index, dtype="object")
    labels[length == 0] = "Empty"
    labels[(length > 0) & (length < 50)] = "Short"
    labels[(length >= 50) & (length <= 200)] = "Medium"
    labels[length > 200] = "Long"
    stats = _counts(labels, total)

    empty_share = _pct(int((labels == "Empty").sum()), total)
    short_share = _pct(int((labels == "Short").sum()), total)

    logger.info(f"column name: {col}")
    logger.info("inferred type: text")
    logger.info(f"missing rate (%%): {empty_share}")
    logger.info("representation slices (bands):")
    _log_slice_stats(stats)

    flags: List[FlagItem] = []
    if empty_share > 10.0:
        flags.append(FlagItem(col, "Empty", f"Empty/missing >10% ({empty_share}%)."))
    if short_share < 5.0:
        flags.append(FlagItem(col, "Short", f"Short reviews <5% ({short_share}%)."))
    if flags:
        logger.info("flagged representation risks:")
        for flag in flags:
            logger.info(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_generic(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    inferred = _infer_type(col, series)
    missing_rate = _pct(int(_missing_mask(series).sum()), total)
    flags: List[FlagItem] = []

    logger.info(f"column name: {col}")
    logger.info(f"inferred type: {inferred}")
    logger.info(f"missing rate (%%): {missing_rate}")
    logger.info("representation slices (bands):")

    if inferred == "numeric":
        num = pd.to_numeric(series, errors="coerce")
        labels = pd.Series(index=num.index, dtype="object")
        labels[num.isna()] = "Missing"
        valid = num.dropna()
        if not valid.empty:
            try:
                q = pd.qcut(valid, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
                labels.loc[q.index] = q.astype(str)
            except ValueError:
                labels.loc[valid.index] = "Q2"
            labels[valid >= valid.quantile(0.99)] = "Outlier (Top 1%)"
        _log_slice_stats(_counts(labels, total))
    elif inferred == "boolean":
        lowered = series.astype(str).str.strip().str.lower()
        labels = pd.Series(index=series.index, dtype="object")
        labels[lowered.isin(BOOL_TRUE)] = "True"
        labels[lowered.isin(BOOL_FALSE)] = "False"
        labels[_missing_mask(series)] = "Missing"
        labels[labels.isna()] = "Invalid"
        _log_slice_stats(_counts(labels, total))
    elif inferred == "text":
        txt = series.astype(str).where(~_missing_mask(series), "")
        labels = pd.Series(index=series.index, dtype="object")
        length = txt.str.len()
        labels[length == 0] = "Empty"
        labels[(length > 0) & (length < 50)] = "Short"
        labels[(length >= 50) & (length <= 200)] = "Medium"
        labels[length > 200] = "Long"
        _log_slice_stats(_counts(labels, total))
    else:
        labels = series.astype(str).str.strip().where(~_missing_mask(series), "Missing")
        stats = _counts(labels, total)
        _log_slice_stats(stats)
        for row in stats:
            if row.label != "Missing" and row.pct < 5.0:
                flags.append(FlagItem(col, row.label, f"Underrepresented slice <5% ({row.pct}%)."))

    if flags:
        logger.info("flagged representation risks:")
        for flag in flags:
            logger.info(f"- {flag.group}: {flag.reason}")
    return flags


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

REVIEW_SAMPLE_SIZE = 100_000  # Bias signals are stable at 100k; loading 2.1M rows OOMs the worker


def _load_review_data(preprocessed_path: str, featured_path: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"Preprocessed file not found: {preprocessed_path}")
    # Stream 2 chunks (100k rows) as a representative sample — avoids OOM on 785MB file.
    # Bias signals (rating distribution, missingness, verified_purchase) are stable at this scale.
    chunks = []
    for chunk in pd.read_json(preprocessed_path, lines=True, chunksize=50_000):
        chunks.append(chunk)
        if sum(len(c) for c in chunks) >= REVIEW_SAMPLE_SIZE:
            break
    df = pd.concat(chunks, ignore_index=True).head(REVIEW_SAMPLE_SIZE)
    logger.info(f"Sampled {len(df)} records (first {REVIEW_SAMPLE_SIZE} rows) for bias analysis")

    if featured_path and os.path.exists(featured_path):
        # Stream featured file with same sample cap to avoid OOM on large featured files.
        feat_chunks = []
        for chunk in pd.read_json(featured_path, lines=True, chunksize=50_000):
            feat_chunks.append(chunk)
            if sum(len(c) for c in feat_chunks) >= REVIEW_SAMPLE_SIZE:
                break
        feat = pd.concat(feat_chunks, ignore_index=True).head(REVIEW_SAMPLE_SIZE)
        # Prefer key-based merge to avoid silent misalignment from index join.
        merge_key_candidates = [["user_id", "asin"], ["user_id", "product_id"]]
        merged = False
        for keys in merge_key_candidates:
            if all(k in df.columns and k in feat.columns for k in keys):
                new_cols = [c for c in feat.columns if c not in df.columns]
                if new_cols:
                    df = df.merge(feat[keys + new_cols], on=keys, how="left")
                merged = True
                break
        if not merged:
            # Fallback: add any columns not already present via index alignment.
            missing_cols = [c for c in feat.columns if c not in df.columns]
            if missing_cols:
                df = df.join(feat[missing_cols])
        logger.info(f"Merged featured data from {featured_path} ({len(df.columns)} total columns)")
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_review_bias(preprocessed_path: str, featured_path: Optional[str] = None) -> None:
    """
    Executes the review bias detection pipeline.
    """
    logger.info("Starting review bias detection...")
    df = _load_review_data(preprocessed_path, featured_path)

    total_rows = len(df)
    total_cols = len(df.columns)

    logger.info(f"\n1) Dataset Summary")
    logger.info("-" * 40)
    logger.info(f"total number of reviews: {total_rows}")
    logger.info(f"number of columns: {total_cols}")
    overall_missing = round(df.isna().sum().sum() / max(df.size, 1) * 100.0, 2)
    logger.info(f"overall missingness rate (%%): {overall_missing}")
    col_missing = {c: _pct(int(_missing_mask(df[c]).sum()), total_rows) for c in df.columns}
    top5 = sorted(col_missing.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("top 5 columns by missingness:")
    for col, pct in top5:
        logger.info(f"- {col}: {pct}%")

    logger.info(f"\n2) Column-by-column analysis")
    logger.info("-" * 40)
    all_flags: List[FlagItem] = []
    for col in df.columns:
        name = _norm(col)
        if name == "user_id":
            all_flags.extend(_profile_user_id(col, df[col], total_rows))
        elif name in {"asin", "product_id"}:
            all_flags.extend(_profile_product_identifier(col, df[col], total_rows))
        elif name == "rating":
            all_flags.extend(_profile_rating(col, df[col], total_rows))
        elif name == "helpful_vote":
            all_flags.extend(_profile_helpful_vote(col, df[col], total_rows))
        elif name == "verified_purchase":
            all_flags.extend(_profile_verified_purchase(col, df[col], total_rows))
        elif name == "review_title":
            all_flags.extend(_profile_review_title(col, df[col], total_rows))
        elif name == "review_text":
            all_flags.extend(_profile_review_text(col, df[col], total_rows))
        else:
            all_flags.extend(_profile_generic(col, df[col], total_rows))

    logger.info(f"\n3) Final Summary")
    logger.info("-" * 40)
    logger.info("all flagged representation risks:")
    if all_flags:
        for flag in all_flags:
            logger.info(f"- {flag.column} | {flag.group}: {flag.reason}")
    else:
        logger.info("- None")

    logger.info("\nrecommended training-time-only mitigations:")
    if all_flags:
        logger.info("- Stratified sampling by rating bucket (negative/neutral/positive), verified_purchase, and per-product review-count band.")
        logger.info("- Controlled oversampling of underrepresented slices (<5%).")
        logger.info("- Evaluation stress tests on negative reviews, neutral reviews, and cold-start products (single-review band).")
        logger.info("- Optional weighting: down-weight helpful_vote==0 if dominance causes overfitting to low-signal reviews.")
    else:
        logger.info("- No flagged representation risks; continue periodic bias monitoring.")

    logger.info("Review bias detection complete.")


if __name__ == "__main__":
    from utils import setup_logging, get_processed_path, get_features_path

    setup_logging()
    run_review_bias(
        preprocessed_path=get_processed_path("review_preprocessed.jsonl"),
        featured_path=get_features_path("review_featured.jsonl"),
    )
