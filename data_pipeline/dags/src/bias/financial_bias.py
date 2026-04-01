"""
Bias Detection for SavVio Financial Data.

Analyzes all columns in financial_preprocessed.csv + financial_featured.csv
for representation gaps, underrepresented groups, and missingness bias.

Input: data/processed/financial_preprocessed.csv
       data/features/financial_featured.csv (optional merge)
Output: Terminal-only log output (no files written)
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

BOOL_TRUE = {"1", "true", "yes", "y", "t"}
BOOL_FALSE = {"0", "false", "no", "n", "f"}


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


# ---------------------------------------------------------------------------
# Low-level helpers (used only in this module)
# ---------------------------------------------------------------------------

def _missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.isna()
    stripped = series.astype(str).str.strip()
    return series.isna() | stripped.eq("")


def _pct(count: int, total: int) -> float:
    return round((count / total * 100.0), 2) if total else 0.0


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _series_lower(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _value_counts_pct(series: pd.Series, total: int, sort_index: bool = False) -> List[SliceStat]:
    counts = series.value_counts(dropna=False, sort=not sort_index)
    if sort_index:
        counts = counts.sort_index()
    return [SliceStat(label=str(k), count=int(v), pct=_pct(int(v), total)) for k, v in counts.items()]


def _normalize_column_name(name: str) -> str:
    return name.strip().lower()


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lookup = {_normalize_column_name(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_column_name(cand)
        if key in lookup:
            return lookup[key]
    return None


def _log_slice_stats(stats: List[SliceStat], indent: str = "  ") -> None:
    for row in stats:
        logger.info(f"{indent}- {row.label}: {row.count} ({row.pct}%)")


# ---------------------------------------------------------------------------
# Column type inference
# ---------------------------------------------------------------------------

def _infer_type(name: str, series: pd.Series) -> str:
    norm_name = _normalize_column_name(name)
    missing = _missing_mask(series)
    non_missing = series[~missing]
    if non_missing.empty:
        if "date" in norm_name or "time" in norm_name:
            return "datetime"
        return "categorical"

    if norm_name == "user_id" or (norm_name.endswith("_id") and "credit" not in norm_name):
        return "id"

    if norm_name == "record_date":
        return "datetime"

    lowered = _series_lower(non_missing)
    if pd.api.types.is_bool_dtype(series) or lowered.isin(BOOL_TRUE | BOOL_FALSE).all():
        return "boolean"

    if "date" in norm_name or "time" in norm_name:
        dt_try = pd.to_datetime(non_missing, errors="coerce")
        if dt_try.notna().mean() > 0.95:
            return "datetime"

    numeric_try = _as_numeric(non_missing)
    if pd.api.types.is_numeric_dtype(series) or numeric_try.notna().mean() >= 0.95:
        return "numeric"

    if norm_name == "job_title":
        return "text"

    uniq_ratio = non_missing.nunique(dropna=True) / max(len(non_missing), 1)
    avg_len = non_missing.astype(str).str.len().mean()
    if uniq_ratio > 0.98:
        return "id"
    if avg_len > 40:
        return "text"
    return "categorical"


# ---------------------------------------------------------------------------
# Numeric banding functions (domain-specific)
# ---------------------------------------------------------------------------

def _band_age(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    labels[(values >= 18) & (values <= 24)] = "Young (18-24)"
    labels[(values >= 25) & (values <= 34)] = "Early-career (25-34)"
    labels[(values >= 35) & (values <= 49)] = "Mid-career (35-49)"
    labels[(values >= 50) & (values <= 64)] = "Late-career (50-64)"
    labels[values >= 65] = "Senior (65+)"
    labels[labels.isna()] = "Out-of-range"
    return labels


def _band_monthly_income(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 3000, 7000, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str)


def _band_monthly_expenses(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 1000, 3000, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str)


def _band_savings(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        [-math.inf, 500, 3000, 15000, math.inf],
        labels=["Near-zero", "Low", "Moderate", "High"],
        right=False,
    ).astype(str)


def _band_liquid_savings(values: pd.Series) -> pd.Series:
    """Liquid savings (feature): same banding as savings; low/near-zero is vulnerable."""
    return pd.cut(
        values,
        [-math.inf, 500, 3000, 15000, math.inf],
        labels=["Near-zero", "Low", "Moderate", "High"],
        right=False,
    ).astype(str)


def _band_emi(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    labels[values == 0] = "None"
    labels[(values > 0) & (values < 500)] = "Low"
    labels[(values >= 500) & (values <= 1500)] = "Moderate"
    labels[values > 1500] = "High"
    return labels


def _band_loan_amount(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 5000, 25000, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str)


def _band_loan_term(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 24, 60, math.inf], labels=["Short", "Medium", "Long"], right=False).astype(str)


def _band_interest(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 5, 12, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str)


def _band_dti(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 0.2, 0.4, math.inf], labels=["Safe", "Warning", "Risky"], right=False).astype(str)


def _band_credit_score(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    labels[values < 580] = "Poor"
    labels[(values >= 580) & (values <= 669)] = "Fair"
    labels[(values >= 670) & (values <= 739)] = "Good"
    labels[(values >= 740) & (values <= 799)] = "Very Good"
    labels[values >= 800] = "Excellent"
    return labels


def _band_savings_income_ratio(values: pd.Series) -> pd.Series:
    return pd.cut(
        values, [-math.inf, 0.25, 1.0, math.inf],
        labels=["Fragile", "Moderate", "Strong"], right=False,
    ).astype(str)


def _band_discretionary_income(values: pd.Series) -> pd.Series:
    return pd.cut(
        values, [-math.inf, 0, 1000, math.inf],
        labels=["Negative", "Tight", "Comfortable"], right=False,
    ).astype(str)


def _band_expense_burden_ratio(values: pd.Series) -> pd.Series:
    return pd.cut(
        values, [-math.inf, 0.5, 0.8, math.inf],
        labels=["Comfortable", "Tight", "Overstretched"], right=False,
    ).astype(str)


def _band_emergency_fund_months(values: pd.Series) -> pd.Series:
    return pd.cut(
        values, [-math.inf, 1, 3, math.inf],
        labels=["Critical", "Fragile", "Stable"], right=False,
    ).astype(str)


def _band_unknown_numeric(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    if values.empty:
        return labels
    try:
        q = pd.qcut(values, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        labels.loc[q.index] = q.astype(str)
    except ValueError:
        labels.loc[values.index] = "Q2"
    top_1 = values.quantile(0.99)
    labels[values >= top_1] = "Outlier (Top 1%)"
    return labels


def _resolve_banding(column: str, values: pd.Series) -> Tuple[pd.Series, Optional[str]]:
    """Return (banded_labels, vulnerable_band_name_or_None)."""
    name = _normalize_column_name(column)

    if name == "age":
        return _band_age(values), None
    if name in {"monthly_income_usd", "monthly_income"}:
        return _band_monthly_income(values), "Low"
    if name in {"monthly_expenses_usd", "monthly_expenses"}:
        return _band_monthly_expenses(values), None
    if name in {"savings_usd", "savings_balance"}:
        return _band_savings(values), "Near-zero"
    if name == "liquid_savings":
        return _band_liquid_savings(values), "Near-zero"
    if name in {"monthly_emi_usd", "monthly_emi"}:
        return _band_emi(values), "High"
    if name in {"loan_amount_usd", "loan_amount"}:
        return _band_loan_amount(values), None
    if name == "loan_term_months":
        return _band_loan_term(values), None
    if name in {"loan_interest_rate_pct", "loan_interest_rate"}:
        return _band_interest(values), None
    if name == "debt_to_income_ratio":
        return _band_dti(values), "Risky"
    if name == "credit_score":
        return _band_credit_score(values), None
    if name in {"savings_to_income_ratio", "saving_to_income_ratio"}:
        return _band_savings_income_ratio(values), None
    # Featured-only columns.
    if name == "discretionary_income":
        return _band_discretionary_income(values), "Negative"
    if name == "monthly_expense_burden_ratio":
        return _band_expense_burden_ratio(values), "Overstretched"
    if name == "emergency_fund_months":
        return _band_emergency_fund_months(values), "Critical"

    return _band_unknown_numeric(values), None


# ---------------------------------------------------------------------------
# Column profilers (one per inferred type)
# ---------------------------------------------------------------------------

def _profile_id(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    missing = int(_missing_mask(series).sum())
    non_missing = series[~_missing_mask(series)]
    uniqueness = (non_missing.nunique(dropna=True) / max(len(non_missing), 1) * 100.0) if len(non_missing) else 0.0

    logger.info(f"Column: {column}")
    logger.info("  - inferred type: id")
    logger.info(f"  - missing rate: {_pct(missing, total_rows)}%")
    logger.info("  - representation slices:")
    _log_slice_stats([SliceStat("Unique (non-missing)", int(non_missing.nunique(dropna=True)), round(uniqueness, 2))])

    flags: List[FlagItem] = []
    if missing > 0:
        flags.append(FlagItem(column, "Missing", f"Identifier has missing values ({_pct(missing, total_rows)}%)."))
    if uniqueness < 95.0:
        flags.append(FlagItem(column, "Uniqueness", f"Uniqueness below 95% ({round(uniqueness, 2)}%)."))

    if flags:
        logger.info("  - flagged groups:")
        for f in flags:
            logger.info(f"    - {f.group}: {f.reason}")
    return flags


def _profile_boolean(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    lowered = _series_lower(series)
    labels = pd.Series(index=series.index, dtype="object")
    labels[lowered.isin(BOOL_TRUE)] = "True"
    labels[lowered.isin(BOOL_FALSE)] = "False"
    labels[_missing_mask(series)] = "Missing"
    labels[labels.isna()] = "Invalid"

    missing = int((labels == "Missing").sum())
    stats = _value_counts_pct(labels, total_rows)

    logger.info(f"Column: {column}")
    logger.info("  - inferred type: boolean")
    logger.info(f"  - missing rate: {_pct(missing, total_rows)}%")
    logger.info("  - representation slices:")
    _log_slice_stats(stats)

    flags: List[FlagItem] = []
    valid = labels[labels.isin(["True", "False"])]
    if not valid.empty:
        valid_dist = valid.value_counts(normalize=True)
        minority_pct = float(valid_dist.min() * 100.0)
        minority_label = str(valid_dist.idxmin())
        if minority_pct < 10.0:
            flags.append(FlagItem(column, minority_label, f"Minority class below 10% ({round(minority_pct, 2)}%)."))
    if flags:
        logger.info("  - flagged groups:")
        for f in flags:
            logger.info(f"    - {f.group}: {f.reason}")
    return flags


def _profile_datetime(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    parsed = pd.to_datetime(series, errors="coerce")
    missing = int(parsed.isna().sum())
    month = parsed.dt.to_period("M").astype(str)
    month = month.where(parsed.notna(), "Missing")
    stats = _value_counts_pct(month, total_rows, sort_index=True)

    logger.info(f"Column: {column}")
    logger.info("  - inferred type: datetime")
    logger.info(f"  - missing rate: {_pct(missing, total_rows)}%")
    logger.info("  - representation slices:")
    _log_slice_stats(stats)

    flags: List[FlagItem] = []
    valid_month = month[month != "Missing"]
    if not valid_month.empty:
        top_share = valid_month.value_counts(normalize=True).max() * 100.0
        top_month = str(valid_month.value_counts(normalize=True).idxmax())
        if top_share > 60.0:
            flags.append(FlagItem(column, top_month, f"One month has >60% of records ({round(top_share, 2)}%)."))

    if flags:
        logger.info("  - flagged groups:")
        for f in flags:
            logger.info(f"    - {f.group}: {f.reason}")
    return flags


def _profile_job_title(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    missing = _missing_mask(series)
    valid = series[~missing].astype(str).str.strip()
    total_valid = len(valid)
    unique_titles = int(valid.nunique(dropna=True))
    top = valid.value_counts().head(10)
    top_total = int(top.sum())
    other_count = max(total_valid - top_total, 0)

    logger.info(f"Column: {column}")
    logger.info("  - inferred type: text")
    logger.info(f"  - missing rate: {_pct(int(missing.sum()), total_rows)}%")
    logger.info("  - representation slices:")
    logger.info(f"  - unique titles: {unique_titles}")
    for title, count in top.items():
        logger.info(f"  - top title '{title}': {count} ({_pct(int(count), total_rows)}%)")
    logger.info(f"  - Other: {other_count} ({_pct(other_count, total_rows)}%)")

    flags: List[FlagItem] = []
    miss_pct = _pct(int(missing.sum()), total_rows)
    if miss_pct > 20.0:
        flags.append(FlagItem(column, "Missing", f"Missing rate above 20% ({miss_pct}%), representation quality risk."))

    if flags:
        logger.info("  - flagged groups:")
        for f in flags:
            logger.info(f"    - {f.group}: {f.reason}")
    return flags


def _profile_categorical(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    labels = series.astype(str).str.strip()
    labels = labels.where(~_missing_mask(series), "Missing")
    stats = _value_counts_pct(labels, total_rows)

    logger.info(f"Column: {column}")
    logger.info("  - inferred type: categorical")
    logger.info(f"  - missing rate: {_pct(int((labels == 'Missing').sum()), total_rows)}%")
    logger.info("  - representation slices:")
    _log_slice_stats(stats)

    flags: List[FlagItem] = []
    for row in stats:
        if row.label != "Missing" and row.pct < 10.0:
            flags.append(FlagItem(column, row.label, f"Category share below 10% ({row.pct}%)."))

    if flags:
        logger.info("  - flagged groups:")
        for f in flags:
            logger.info(f"    - {f.group}: {f.reason}")
    return flags


def _profile_numeric(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    numeric = _as_numeric(series)
    missing = int(numeric.isna().sum())
    valid = numeric.dropna()
    bands = pd.Series(index=numeric.index, dtype="object")
    vulnerable_band: Optional[str]
    if not valid.empty:
        assigned, vulnerable_band = _resolve_banding(column, valid)
        bands.loc[valid.index] = assigned.astype(str)
    else:
        vulnerable_band = None

    bands.loc[numeric.isna()] = "Missing"
    stats = _value_counts_pct(bands, total_rows)

    logger.info(f"Column: {column}")
    logger.info("  - inferred type: numeric")
    logger.info(f"  - missing rate: {_pct(missing, total_rows)}%")
    logger.info("  - representation slices:")
    _log_slice_stats(stats)

    flags: List[FlagItem] = []
    if vulnerable_band is not None:
        vuln_count = int((bands == vulnerable_band).sum())
        vuln_pct = _pct(vuln_count, total_rows)
        if vuln_count == 0 or vuln_pct < 10.0:
            flags.append(FlagItem(
                column, vulnerable_band,
                f"Vulnerable/high-risk band underrepresented ({vuln_pct}%; threshold 10%).",
            ))

    if flags:
        logger.info("  - flagged groups:")
        for f in flags:
            logger.info(f"    - {f.group}: {f.reason}")
    return flags


# ---------------------------------------------------------------------------
# Cross-column missingness bias
# ---------------------------------------------------------------------------

def _apply_missingness_bias_checks(df: pd.DataFrame) -> List[FlagItem]:
    flags: List[FlagItem] = []
    key_numeric_candidates = [
        "monthly_income_usd", "monthly_expenses_usd", "savings_usd",
        "monthly_emi_usd", "loan_amount_usd",
        "debt_to_income_ratio", "credit_score", "savings_to_income_ratio",
        # legacy / featured aliases
        "monthly_income", "monthly_expenses", "savings_balance",
        "monthly_emi", "loan_amount", "saving_to_income_ratio",
        # featured-only columns
        "discretionary_income", "monthly_expense_burden_ratio", "emergency_fund_months",
    ]
    numeric_cols: List[str] = []
    for name in key_numeric_candidates:
        resolved = _first_existing(df, [name])
        if resolved and resolved not in numeric_cols:
            numeric_cols.append(resolved)

    categorical_targets = ["gender", "education_level", "employment_status", "loan_type", "region"]
    cat_cols = [c for c in [_first_existing(df, [n]) for n in categorical_targets] if c is not None]

    for num_col in numeric_cols:
        overall_missing = _missing_mask(df[num_col]).mean()
        if overall_missing <= 0:
            continue
        for cat_col in cat_cols:
            groups = df[cat_col].astype(str).str.strip().where(~_missing_mask(df[cat_col]), "Missing")
            by_group = _missing_mask(df[num_col]).groupby(groups).mean()
            for grp, rate in by_group.items():
                if rate > (2.0 * overall_missing):
                    flags.append(FlagItem(
                        num_col, f"{cat_col}={grp}",
                        f"Missingness bias: subgroup missing {round(rate * 100.0, 2)}% "
                        f">2x overall {round(overall_missing * 100.0, 2)}%.",
                    ))
    return flags


# ---------------------------------------------------------------------------
# Cross-column: EMI exceeds monthly income (debt burden bias)
# ---------------------------------------------------------------------------

def _apply_emi_exceeds_income_check(df: pd.DataFrame) -> List[FlagItem]:
    """
    Flag when EMI (monthly loan payment) is greater than monthly income.
    That indicates unsustainable debt burden and is a bias/vulnerability to detect.
    """
    flags: List[FlagItem] = []
    income_col = _first_existing(df, ["monthly_income", "monthly_income_usd"])
    emi_col = _first_existing(df, ["monthly_emi", "monthly_emi_usd"])
    if income_col is None or emi_col is None:
        return flags
    income = _as_numeric(df[income_col])
    emi = _as_numeric(df[emi_col])
    # Only consider rows with positive income and non-missing EMI
    valid = income.notna() & emi.notna() & (income > 0)
    emi_exceeds_income = valid & (emi > income)
    n = int(emi_exceeds_income.sum())
    total_valid = int(valid.sum())
    if total_valid == 0:
        return flags
    pct = _pct(n, total_valid)
    logger.info(f"\nCross-column check: EMI vs income")
    logger.info("-" * 40)
    logger.info(f"  Rows with EMI > monthly_income: {n} ({pct}% of rows with valid income and EMI)")
    if n > 0:
        flags.append(
            FlagItem(
                "monthly_emi vs monthly_income",
                "EMI exceeds income",
                f"Debt burden bias: {n} rows ({pct}%) have EMI greater than monthly income — unsustainable; review data or treat as vulnerable slice.",
            )
        )
    return flags


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dataset(processed_path: str, featured_path: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Required file not found: {processed_path}")

    df = pd.read_csv(processed_path)
    logger.info(f"Loaded {len(df)} records from {processed_path}")

    if featured_path and os.path.exists(featured_path):
        feat = pd.read_csv(featured_path)
        join_key = _first_existing(df, ["user_id"])
        feat_key = _first_existing(feat, ["user_id"])
        if join_key and feat_key:
            feat_no_overlap = feat.drop(
                columns=[c for c in feat.columns if c in df.columns and c != feat_key],
                errors="ignore",
            )
            df = df.merge(feat_no_overlap, left_on=join_key, right_on=feat_key, how="left")
            if feat_key != join_key and feat_key in df.columns:
                df = df.drop(columns=[feat_key])
        logger.info(f"Merged featured data from {featured_path} ({len(df.columns)} total columns)")
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_financial_bias(processed_path: str, featured_path: Optional[str] = None) -> None:
    """
    Executes the financial bias detection pipeline.
    """
    logger.info("Starting financial bias detection...")
    df = _load_dataset(processed_path, featured_path)
    total_rows = len(df)
    total_cols = len(df.columns)

    logger.info(f"\nDataset summary:")
    logger.info("-" * 40)
    overall_missing = float(df.isna().sum().sum() / max(df.size, 1) * 100.0)
    logger.info(f"rows: {total_rows}, columns: {total_cols}")
    logger.info(f"overall missingness (%%): {round(overall_missing, 2)}")

    col_missing = {col: _pct(int(_missing_mask(df[col]).sum()), total_rows) for col in df.columns}
    top5 = sorted(col_missing.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("top 5 columns by missingness:")
    for col, pct in top5:
        logger.info(f"- {col}: {pct}%")

    logger.info(f"\nColumn-level representation analysis:")
    logger.info("-" * 40)
    all_flags: List[FlagItem] = []
    for column in df.columns:
        series = df[column]
        inferred = _infer_type(column, series)
        norm_col = _normalize_column_name(column)

        if norm_col == "user_id":
            all_flags.extend(_profile_id(column, series, total_rows))
            continue
        if norm_col == "record_date" or inferred == "datetime":
            all_flags.extend(_profile_datetime(column, series, total_rows))
            continue
        if norm_col == "job_title":
            all_flags.extend(_profile_job_title(column, series, total_rows))
            continue
        if norm_col == "has_loan" or inferred == "boolean":
            all_flags.extend(_profile_boolean(column, series, total_rows))
            continue
        if inferred == "numeric":
            all_flags.extend(_profile_numeric(column, series, total_rows))
            continue
        if inferred == "id":
            all_flags.extend(_profile_id(column, series, total_rows))
            continue
        all_flags.extend(_profile_categorical(column, series, total_rows))

    all_flags.extend(_apply_missingness_bias_checks(df))
    all_flags.extend(_apply_emi_exceeds_income_check(df))

    logger.info(f"\nFlagged Representation Risks")
    logger.info("-" * 40)
    if not all_flags:
        logger.info("- None")
    else:
        for f in all_flags:
            logger.info(f"- {f.column} | {f.group}: {f.reason}")

    logger.info(f"\nRecommended Training-Time Mitigations")
    logger.info("-" * 40)
    if not all_flags:
        logger.info("- No representation risks were flagged; continue standard monitoring.")
    else:
        logger.info("- Stratified sampling by flagged bands/groups.")
        logger.info("- Controlled oversampling of underrepresented high-risk bands (training only).")
        if any(_normalize_column_name(f.column) == "record_date" for f in all_flags):
            logger.info("- Time-stratified train/validation split due to temporal skew.")

    logger.info("Financial bias detection complete.")


if __name__ == "__main__":
    from utils import setup_logging, get_processed_path, get_features_path

    setup_logging()
    run_financial_bias(
        processed_path=get_processed_path("financial_preprocessed.csv"),
        featured_path=get_features_path("financial_featured.csv"),
    )
