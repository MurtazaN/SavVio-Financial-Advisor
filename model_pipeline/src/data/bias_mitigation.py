"""
Pre-Training Bias Mitigation for SavVio Model Pipeline.

Called from training_data_generator.py BEFORE scenario sampling.
Also used by run_pipeline.py via apply_all_mitigations().

Addresses every flag raised by the bias detection pipeline:

FINANCIAL FLAGS:
  - savings_balance Near-zero: 0.0%  → synthetic augmentation + oversample
  - liquid_savings  Near-zero: 0.0%  → derived from savings_balance; fixed upstream
  - employment_status Unemployed/Student: ~10%  → oversample to ≥12%
  - EMI > monthly_income: 23.99%     → triage + conditional keep/flag

PRODUCT FLAGS:
  - category: 80+ leaf nodes all <5% → collapse to 3-level taxonomy
  - rating_number Low-confidence: 44.6% dominance → confidence-band stratify
  - Premium price: 3.96%             → oversample premium tier

REVIEW FLAGS:
  - rating Neutral: 4.89%            → oversample to ≥7%
  - verified_purchase False: 4.16%   → oversample to ≥7%
  - helpful_vote None: 80.8%         → sample weight (not oversample)
  - user_id uniqueness 83.4%         → per-user review cap

Usage:
    from data.bias_mitigation import apply_all_mitigations

    fin_df, prod_df, rev_df = apply_all_mitigations(fin_df, prod_df, rev_df)
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target minimum representation after mitigation (as fraction of dataset).
MIN_EMPLOYMENT_SHARE = 0.12    # Unemployed and Student each
MIN_NEUTRAL_SHARE    = 0.07    # Neutral ratings
MIN_UNVERIFIED_SHARE = 0.07    # Unverified purchases
MIN_PREMIUM_SHARE    = 0.06    # Premium products

# EMI sanity: flag but keep rows where EMI > income (vulnerable slice).
# Only drop if EMI is clearly erroneous (e.g. 10x income — data entry error).
EMI_HARD_DROP_MULTIPLIER = 10.0

# Per-user review cap: prevents prolific reviewers from dominating.
MAX_REVIEWS_PER_USER = 50

# Random seed for all sampling operations — must match Config.RANDOM_STATE.
RANDOM_STATE = 42

# Near-zero savings: define as savings_balance < $200.
NEAR_ZERO_SAVINGS_THRESHOLD = 200.0


def _safe_share(df: pd.DataFrame, col: str, mask_fn) -> float:
    """Return masked share in [0,1], safely handling empty/missing columns."""
    if df.empty or col not in df.columns:
        return 0.0
    mask = mask_fn(df[col])
    return float(mask.sum()) / float(len(df)) if len(df) else 0.0

# ---------------------------------------------------------------------------
# Stage 1: Financial data repair + rebalancing
# ---------------------------------------------------------------------------

def repair_emi_anomalies(fin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Triage EMI > monthly_income rows:
      - EMI > 10× income  → zero out EMI (data entry error — not dropped,
                            user profile is preserved with corrected EMI)
      - EMI > income but ≤ 10×  → legitimate vulnerable slice,
                                   add over_leveraged_flag = 1

    Robust to missing monthly_emi or monthly_income columns.
    """
    if "monthly_income" not in fin_df.columns or "monthly_emi" not in fin_df.columns:
        logger.info("repair_emi_anomalies: missing income or EMI column — skipping.")
        fin_df["over_leveraged_flag"] = 0
        return fin_df

    income = fin_df["monthly_income"]
    emi    = fin_df["monthly_emi"]

    hard_error = (emi > 0) & (income > 0) & (emi > EMI_HARD_DROP_MULTIPLIER * income)
    n_errors = int(hard_error.sum())
    if n_errors:
        logger.warning(
            "Zeroing monthly_emi for %d rows where EMI > %.0f× income (data entry error).",
            n_errors, EMI_HARD_DROP_MULTIPLIER,
        )
        fin_df.loc[hard_error, "monthly_emi"] = 0.0

    over_leveraged = (fin_df["monthly_emi"] > 0) & (income > 0) & (fin_df["monthly_emi"] > income)
    fin_df["over_leveraged_flag"] = over_leveraged.astype(int)
    logger.info(
        "Flagged %d over-leveraged rows (EMI > income) as vulnerable slice.",
        int(over_leveraged.sum()),
    )
    return fin_df


def augment_near_zero_savings(
    fin_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    The bias detector found 0 users with Near-zero savings_balance (<$200).
    This is almost certainly a Kaggle dataset artifact — real populations
    always contain people with minimal liquid savings.

    Strategy: synthesise a small cohort (≈3% of dataset) by:
      1. Sampling from the Low-income band (monthly_income < $3,000).
      2. Replacing savings_balance with a draw from Uniform(0, 200).
      3. Recalculating liquid_savings downstream (financial_features.py
         will re-derive it; we set a placeholder of 0 here).

    NOTE: Synthetic rows get synthetic_flag=1 so they can be identified
    and excluded from any held-out evaluation set.
    """
    if fin_df.empty:
        logger.info("augment_near_zero_savings: empty DataFrame — skipping.")
        return fin_df
    if "monthly_income" not in fin_df.columns:
        logger.info("augment_near_zero_savings: missing monthly_income column — skipping.")
        return fin_df

    target_n = max(int(len(fin_df) * 0.03), 500)
    low_income = fin_df[fin_df["monthly_income"] < 3_000]

    if len(low_income) == 0:
        logger.warning("No low-income users found — cannot augment Near-zero savings.")
        return fin_df

    sampled = low_income.sample(
        n=target_n, replace=True, random_state=int(rng.integers(0, 2**31))
    )
    sampled = sampled.copy()
    sampled["savings_balance"] = rng.uniform(0, NEAR_ZERO_SAVINGS_THRESHOLD, size=len(sampled))
    sampled["liquid_savings"]  = sampled["savings_balance"] * 0.90  # near-full liquidity at this level
    sampled["synthetic_flag"]  = 1

    fin_df["synthetic_flag"] = 0
    augmented = pd.concat([fin_df, sampled], ignore_index=True)
    logger.info(
        "Augmented dataset with %d synthetic Near-zero savings rows (%.1f%% of original).",
        target_n, target_n / len(fin_df) * 100,
    )
    return augmented


def oversample_minority_employment(
    fin_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Oversample Unemployed and Student status groups to ≥ MIN_EMPLOYMENT_SHARE each.

    Fix: recompute total after each group is added so the second group's
    target is computed against the already-updated population size, not
    the stale pre-loop total.
    """
    if fin_df.empty:
        logger.info("oversample_minority_employment: empty DataFrame — skipping.")
        return fin_df
    if "employment_status" not in fin_df.columns:
        logger.info("oversample_minority_employment: missing employment_status column — skipping.")
        return fin_df

    for status in ["Unemployed", "Student"]:
        # Recompute total inside the loop — fixes stale denominator bug.
        total = len(fin_df)
        group = fin_df[fin_df["employment_status"] == status]
        current_share = len(group) / total

        if current_share >= MIN_EMPLOYMENT_SHARE:
            logger.info(
                "employment_status='%s' already at %.1f%% — no oversampling needed.",
                status, current_share * 100,
            )
            continue

        target_n = int(MIN_EMPLOYMENT_SHARE * total) - len(group)
        if target_n <= 0:
            continue
        if group.empty:
            logger.warning(
                "oversample_minority_employment: no rows found for '%s' — cannot oversample.",
                status,
            )
            continue

        extra = group.sample(
            n=target_n, replace=True,
            random_state=int(rng.integers(0, 2**31)),
        )
        extra = extra.copy()
        extra["synthetic_flag"] = extra.get("synthetic_flag", pd.Series(0, index=extra.index))
        fin_df = pd.concat([fin_df, extra], ignore_index=True)
        new_share = len(fin_df[fin_df["employment_status"] == status]) / len(fin_df)
        logger.info(
            "Oversampled '%s' by %d rows: %.1f%% → %.1f%%.",
            status, target_n, current_share * 100, new_share * 100,
        )

    return fin_df


# ---------------------------------------------------------------------------
# Stage 2: Product data repair + rebalancing
# ---------------------------------------------------------------------------

def collapse_category_taxonomy(prod_df: pd.DataFrame) -> pd.DataFrame:
    """
    The bias detector found 80+ leaf-level categories, each <5%.
    At this granularity every category is flagged as underrepresented —
    the threshold is not meaningful.

    Strategy: collapse to the first 3 path components.
    E.g. "Appliances > Parts & Accessories > Refrigerator Parts & Accessories > Ice Makers"
      → "Appliances > Parts & Accessories > Refrigerator Parts & Accessories"

    This reduces to ~15 meaningful top-level categories where representation
    flags carry real signal.

    The original path is preserved in `category_leaf` for any downstream
    use that needs it.
    """
    if "category" not in prod_df.columns:
        return prod_df

    prod_df["category_leaf"] = prod_df["category"]
    prod_df["category"] = (
        prod_df["category"]
        .astype(str)
        .str.split(" > ")
        .apply(lambda parts: " > ".join(parts[:3]) if isinstance(parts, list) else parts)
    )

    n_collapsed = prod_df["category"].nunique(dropna=True)
    logger.info("Collapsed product categories from 80+ leaf nodes to %d top-level groups.", n_collapsed)
    return prod_df


def oversample_premium_products(
    prod_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Premium products (price > $200) are at 3.96%, below the 5% threshold.
    Oversample premium tier to MIN_PREMIUM_SHARE.

    Note: oversampling happens on the products table, which means premium
    products get sampled more often during scenario pairing in
    training_data_generator.py. We add a weight column rather than
    duplicating rows to avoid inflating the product catalogue.
    """
    if prod_df.empty:
        logger.info("oversample_premium_products: empty DataFrame — skipping.")
        return prod_df
    if "price" not in prod_df.columns:
        logger.info("oversample_premium_products: missing price column — skipping.")
        prod_df["sample_weight"] = 1.0
        return prod_df

    premium_mask = prod_df["price"] > 200
    premium_share = premium_mask.sum() / len(prod_df)

    if premium_share >= MIN_PREMIUM_SHARE:
        prod_df["sample_weight"] = 1.0
        return prod_df

    # Assign upweighted sample_weight to premium products.
    # The stratified sampler in training_data_generator.py uses this weight.
    target_weight = MIN_PREMIUM_SHARE / max(premium_share, 1e-6)
    prod_df["sample_weight"] = np.where(premium_mask, target_weight, 1.0)
    logger.info(
        "Premium products upweighted %.2f× (%.1f%% → target %.1f%%).",
        target_weight, premium_share * 100, MIN_PREMIUM_SHARE * 100,
    )
    return prod_df


# ---------------------------------------------------------------------------
# Stage 3: Review data repair + rebalancing
# ---------------------------------------------------------------------------

def cap_prolific_reviewers(rev_df: pd.DataFrame) -> pd.DataFrame:
    """
    user_id uniqueness is 83.4% — a meaningful tail of users has written
    many reviews. Without capping, training over-indexes on their preferences.

    Strategy: keep at most MAX_REVIEWS_PER_USER reviews per user,
    sampled randomly (so no single product is always included or excluded).

    Robust to empty DataFrame or missing user_id column.
    """
    if rev_df.empty or "user_id" not in rev_df.columns:
        logger.info("cap_prolific_reviewers: empty DataFrame or missing user_id — skipping.")
        return rev_df

    before = len(rev_df)
    # We want "at most MAX_REVIEWS_PER_USER per user".
    # pandas' groupby().sample(n=..., replace=False) crashes if a group has fewer
    # than n rows, so we:
    #  - keep all rows for users with count <= MAX_REVIEWS_PER_USER
    #  - sample exactly MAX_REVIEWS_PER_USER for users with count > MAX_REVIEWS_PER_USER
    user_counts = rev_df["user_id"].value_counts(dropna=False)
    big_users = user_counts[user_counts > MAX_REVIEWS_PER_USER].index
    small_or_equal_users = user_counts[user_counts <= MAX_REVIEWS_PER_USER].index

    # Take all reviews for small users.
    rev_small = rev_df[rev_df["user_id"].isin(small_or_equal_users)]
    # Randomly sample MAX_REVIEWS_PER_USER for large users only.
    rev_big = (
        rev_df[rev_df["user_id"].isin(big_users)]
        .groupby("user_id", group_keys=False)
        .sample(n=MAX_REVIEWS_PER_USER, replace=False, random_state=RANDOM_STATE)
    )

    rev_df = pd.concat([rev_small, rev_big], ignore_index=True)
    after = len(rev_df)
    logger.info(
        "Reviewer cap (max %d/user): %d → %d reviews (removed %d).",
        MAX_REVIEWS_PER_USER, before, after, before - after,
    )
    return rev_df


def oversample_minority_ratings(
    rev_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Neutral ratings (3 stars) are at 4.89%, just below the flagged threshold.
    Oversample to MIN_NEUTRAL_SHARE so the model sees enough fence-sitters.
    """
    if rev_df.empty:
        logger.info("oversample_minority_ratings: empty DataFrame — skipping.")
        return rev_df
    if "rating" not in rev_df.columns:
        logger.info("oversample_minority_ratings: missing rating column — skipping.")
        return rev_df

    total = len(rev_df)
    neutral = rev_df[rev_df["rating"] == 3]
    current_share = len(neutral) / total

    if current_share >= MIN_NEUTRAL_SHARE:
        logger.info("Neutral ratings already at %.1f%% — no oversampling needed.", current_share * 100)
        return rev_df

    target_n = int(MIN_NEUTRAL_SHARE * total) - len(neutral)
    if target_n <= 0:
        return rev_df
    if neutral.empty:
        logger.warning("oversample_minority_ratings: no neutral rows found — cannot oversample.")
        return rev_df
    extra = neutral.sample(
        n=target_n, replace=True, random_state=int(rng.integers(0, 2**31))
    )
    rev_df = pd.concat([rev_df, extra], ignore_index=True)
    logger.info(
        "Oversampled Neutral ratings by %d rows: %.1f%% → %.1f%%.",
        target_n, current_share * 100, MIN_NEUTRAL_SHARE * 100,
    )
    return rev_df


def oversample_unverified_purchases(
    rev_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Unverified purchases are at 4.16%. These may carry a different signal
    (gifted products, repeat buyers) and must be present for the model to
    learn the verified_purchase feature meaningfully.
    """
    if rev_df.empty:
        logger.info("oversample_unverified_purchases: empty DataFrame — skipping.")
        return rev_df
    if "verified_purchase" not in rev_df.columns:
        logger.info("oversample_unverified_purchases: missing verified_purchase column — skipping.")
        return rev_df

    total = len(rev_df)

    # Normalise the column to bool regardless of stored type.
    verified_col = rev_df["verified_purchase"].astype(str).str.strip().str.lower()
    unverified_mask = verified_col.isin({"0", "false", "no", "n", "f"})
    unverified = rev_df[unverified_mask]
    current_share = len(unverified) / total

    if current_share >= MIN_UNVERIFIED_SHARE:
        logger.info("Unverified purchases already at %.1f%% — no oversampling needed.", current_share * 100)
        return rev_df

    target_n = int(MIN_UNVERIFIED_SHARE * total) - len(unverified)
    if target_n <= 0:
        return rev_df
    if unverified.empty:
        logger.warning("oversample_unverified_purchases: no unverified rows found — cannot oversample.")
        return rev_df
    extra = unverified.sample(
        n=target_n, replace=True, random_state=int(rng.integers(0, 2**31))
    )
    rev_df = pd.concat([rev_df, extra], ignore_index=True)
    logger.info(
        "Oversampled unverified purchases by %d rows: %.1f%% → %.1f%%.",
        target_n, current_share * 100, MIN_UNVERIFIED_SHARE * 100,
    )
    return rev_df


def compute_review_sample_weights(rev_df: pd.DataFrame) -> pd.DataFrame:
    """
    80.8% of reviews have helpful_vote=0 — if we oversample these,
    we just add more low-signal noise. Instead, assign inverse-frequency
    sample weights:
      - helpful_vote > 10 → weight 4.0  (highly endorsed reviews)
      - helpful_vote 3–10 → weight 2.0
      - helpful_vote 1–2  → weight 1.5
      - helpful_vote = 0  → weight 1.0 (baseline)

    These weights feed into XGBoost/LightGBM sample_weight parameter.
    """
    if rev_df.empty:
        logger.info("compute_review_sample_weights: empty DataFrame — skipping.")
        rev_df["sample_weight"] = pd.Series(dtype=float)
        return rev_df

    helpful_raw = rev_df["helpful_vote"] if "helpful_vote" in rev_df.columns else pd.Series(0, index=rev_df.index)
    helpful = pd.to_numeric(helpful_raw, errors="coerce").fillna(0)
    weights = pd.Series(1.0, index=rev_df.index)
    weights[helpful >= 1]  = 1.5
    weights[helpful >= 3]  = 2.0
    weights[helpful >= 10] = 4.0
    rev_df["sample_weight"] = weights
    logger.info(
        "Assigned helpful_vote sample weights: "
        "mean=%.2f, weight>1 covers %.1f%% of reviews.",
        weights.mean(),
        (weights > 1.0).sum() / len(weights) * 100,
    )
    return rev_df


# ---------------------------------------------------------------------------
# Label distribution guard (Stage 7 in pipeline)
# ---------------------------------------------------------------------------

def assert_label_balance(scenarios: pd.DataFrame, label_col: str = "final_recommendation") -> None:
    """
    Assert that no label class is below 15% before the train/val/test split.
    Raises ValueError if violated — the pipeline must not proceed with a
    severely imbalanced label set.

    Called from run_pipeline.py after generate_training_data().
    """
    if label_col not in scenarios.columns:
        logger.warning("Label column '%s' not found — skipping label balance check.", label_col)
        return

    dist = scenarios[label_col].value_counts(normalize=True)
    for label, share in dist.items():
        if share < 0.15:
            raise ValueError(
                f"Label balance check FAILED: '{label}' is {share:.1%} of training data "
                f"(minimum required: 15%). "
                f"Adjust the deterministic engine thresholds or increase scenario sampling."
            )
    logger.info("Label balance check passed: %s", dict(dist.round(3)))


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def apply_all_mitigations(
    fin_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    rev_df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply all pre-training bias mitigations in the correct order.

    Call this from training_data_generator.py before generate_scenarios().

    Args:
        fin_df:  financial_profiles DataFrame from PostgreSQL.
        prod_df: products DataFrame from PostgreSQL.
        rev_df:  reviews DataFrame from PostgreSQL.

    Returns:
        (fin_df, prod_df, rev_df) with mitigations applied in-place copies.
    """
    rng = np.random.default_rng(random_state)

    logger.info("=" * 60)
    logger.info("APPLYING PRE-TRAINING BIAS MITIGATIONS")
    logger.info("=" * 60)
    logger.info(
        "Before mitigation shares — unemployed: %.2f%%, student: %.2f%%, premium_products: %.2f%%, neutral_reviews: %.2f%%, unverified_reviews: %.2f%%",
        _safe_share(fin_df, "employment_status", lambda s: s == "Unemployed") * 100,
        _safe_share(fin_df, "employment_status", lambda s: s == "Student") * 100,
        _safe_share(prod_df, "price", lambda s: s > 200) * 100,
        _safe_share(rev_df, "rating", lambda s: s == 3) * 100,
        _safe_share(
            rev_df,
            "verified_purchase",
            lambda s: s.astype(str).str.strip().str.lower().isin({"0", "false", "no", "n", "f"}),
        ) * 100,
    )

    # --- Financial ---
    logger.info("--- Financial mitigations ---")
    fin_df = repair_emi_anomalies(fin_df)
    fin_df = augment_near_zero_savings(fin_df, rng)
    fin_df = oversample_minority_employment(fin_df, rng)

    # --- Product ---
    logger.info("--- Product mitigations ---")
    prod_df = collapse_category_taxonomy(prod_df)
    prod_df = oversample_premium_products(prod_df, rng)

    # --- Review ---
    logger.info("--- Review mitigations ---")
    rev_df = cap_prolific_reviewers(rev_df)
    rev_df = oversample_minority_ratings(rev_df, rng)
    rev_df = oversample_unverified_purchases(rev_df, rng)
    rev_df = compute_review_sample_weights(rev_df)

    logger.info("Pre-training bias mitigations complete.")
    logger.info(
        "After mitigation shares — unemployed: %.2f%%, student: %.2f%%, premium_products: %.2f%%, neutral_reviews: %.2f%%, unverified_reviews: %.2f%%",
        _safe_share(fin_df, "employment_status", lambda s: s == "Unemployed") * 100,
        _safe_share(fin_df, "employment_status", lambda s: s == "Student") * 100,
        _safe_share(prod_df, "price", lambda s: s > 200) * 100,
        _safe_share(rev_df, "rating", lambda s: s == 3) * 100,
        _safe_share(
            rev_df,
            "verified_purchase",
            lambda s: s.astype(str).str.strip().str.lower().isin({"0", "false", "no", "n", "f"}),
        ) * 100,
    )
    logger.info(
        "Final sizes — financial: %d, products: %d, reviews: %d",
        len(fin_df), len(prod_df), len(rev_df),
    )
    return fin_df, prod_df, rev_df