# Deterministic Decision Engine — Layer 1 (Financial) & Layer 2 (Product/Review)

The decision engine is a two-layer rule-based system that labels every (user, product) pair as
GREEN, YELLOW, or RED. It is **authoritative** — neither the ML model nor the LLM wrapper may
override its output.

Source files:
- `model_pipeline/src/deterministic_engine/financial_engine.py` (Layer 1)
- `model_pipeline/src/deterministic_engine/downgrade_engine.py` (Layer 2)

---

## Overview

```
User + Product
     │
     ▼
┌──────────────────────────────┐
│  Layer 1: Financial Engine   │  Uses 11 financial features (5 from DB + 6 computed).
│  (financial_engine.py)       │  Evaluates 4 RED rules, then 5 YELLOW rules.
│                              │  Output: GREEN / YELLOW / RED
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Layer 2: Downgrade Engine   │  Uses 7 product features + 6 review features.
│  (downgrade_engine.py)       │  Can only downgrade by 1 step (GREEN→YELLOW, YELLOW→RED).
│                              │  Requires BOTH product AND review rules to fire.
└──────────────┬───────────────┘
               │
               ▼
         Final Label
    (GREEN / YELLOW / RED)
```

**Label meanings:**
- **GREEN** — Purchase fits comfortably within the user's financial life.
- **YELLOW** — Genuine concern — the user should pause and think.
- **RED** — Purchase would cause financial harm.

---

## Layer 1: Financial Engine (`financial_engine.py`)

### Design principle

Every rule combines signals from **at least 2 independent correlation groups** to avoid false
triggers from a single underlying cause. For example, low savings alone doesn't trigger RED —
it must be combined with unaffordable income AND a non-trivial purchase.

### Correlation groups

| Group | What it measures | Features |
|-------|-----------------|----------|
| **Group 1** — Income capacity | Can the user's monthly cash flow handle this? | `affordability_score`, `discretionary_income`, `price_to_income_ratio` |
| **Group 2** — Savings depth | Does the user have a financial safety net? | `saving_to_income_ratio`, `savings_to_price_ratio`, `emergency_fund_months`, `residual_utility_score` |
| **Group 3** — Debt burden | How stretched are the user's obligations? | `debt_to_income_ratio`, `monthly_expense_burden_ratio` |
| **Group 4** — Independent signals | Credit and net worth health | `credit_risk_indicator`, `net_worth_indicator` |

### Features used (11 total)

**5 from database** (pre-computed by data pipeline):
- `discretionary_income` — monthly income minus all recurring expenses
- `debt_to_income_ratio` — total debt payments / income
- `saving_to_income_ratio` — savings / income
- `monthly_expense_burden_ratio` — (expenses + EMI) / income
- `emergency_fund_months` — savings / (expenses + EMI)

**6 computed per (user, product) pair** (see `feature_computation_and_training_data.md`):
- `affordability_score` = discretionary_income − product_price
- `price_to_income_ratio` (PIR) = product_price / monthly_income
- `residual_utility_score` (RUS) = (savings − price) / total_obligations
- `savings_to_price_ratio` (SPR) = savings / product_price
- `net_worth_indicator` (NWI) = (savings − loan_amount) / income
- `credit_risk_indicator` (CRI) = (credit_score − 299) / 550

### RED rules (any single rule triggers RED)

RED means the purchase would cause financial harm. Each rule is a compound AND condition
that crosses at least 2 correlation groups. Each rule includes a **PIR escape hatch** so that
trivially cheap purchases (low PIR) never trigger RED.

| Rule | Conditions | Groups crossed | Explanation |
|------|-----------|---------------|-------------|
| **RED 1** | `affordability < 0` AND `SPR < 1.5` AND `PIR > 0.10` | 1 + 2 | Income can't handle it, savings barely cover the price, and the purchase is non-trivial. |
| **RED 2** | `MEB > 0.80` AND `PIR > 0.20` AND `EFM < 3.0` | 3 + 1 + 2 | Budget over 80% consumed, product is 20%+ of income, and thin emergency fund. |
| **RED 3** | `NWI < −2.0` AND `affordability < 0` AND `PIR > 0.15` AND `EFM < 3.0` | 4 + 1 + 2 | Net worth deeply negative, no surplus income, and a significant purchase. |
| **RED 4** | `EFM < 1.0` AND `DTI > 0.30` AND `PIR > 0.10` | 2 + 3 + 1 | Less than 1 month emergency fund, DTI over 30%, and purchase is non-trivial. |

**PIR escape hatch:** Every RED rule requires PIR above a threshold (0.10–0.20). If a user is
financially stressed but the product costs $5, it won't trigger RED. This prevents the engine
from blocking trivial purchases for users who happen to have poor finances.

### YELLOW rules (2+ rules must trigger for YELLOW)

YELLOW means genuine concern — the user should pause and think. Unlike RED (where any single
rule is sufficient), YELLOW requires **at least 2 rules to fire simultaneously**.

| Rule | Conditions | Groups crossed | Explanation |
|------|-----------|---------------|-------------|
| **YEL 1** | `affordability < 0` AND `PIR > 0.25` AND `SPR < 5.0` | 1 + 2 | Income doesn't cover purchase directly, and savings cushion is moderate. |
| **YEL 2** | `SPR < 5.0` AND `PIR > 0.10` | 2 + 1 | Savings can't comfortably absorb the purchase. |
| **YEL 3** | `MEB > 0.70` AND `PIR > 0.10` AND `SPR < 5.0` | 3 + 1 + 2 | High debt obligations coupled with significant purchase and thin savings. |
| **YEL 4** | `EFM < 3.0` AND `affordability < 0` | 2 + 1 | Low emergency funds for an unaffordable immediate expense. |
| **YEL 5** | `CRI < 0.35` AND `NWI < 1.0` AND `PIR > 0.15` AND `SPR < 5.0` | 4 + 1 + 2 | Weak overall financial profile for a significant purchase. |

### GREEN (default)

If no RED rules fire AND fewer than 2 YELLOW rules fire, the label is GREEN.

### Evaluation order

```
1. Evaluate all 4 RED rules (short-circuit: first RED rule that fires → return RED)
2. Evaluate all 5 YELLOW rules (count how many fire)
3. If 2+ YELLOW rules fired → return YELLOW
4. Otherwise → return GREEN
```

### API

**Single pair (inference time):**
```python
from deterministic_engine.financial_engine import DecisionEngine

engine = DecisionEngine()
result = engine.decide(
    financial={"affordability_score": -500, "price_to_income_ratio": 0.25, ...},
    product={"price": 1200},
)
# result.decision_category → "RED"
# result.triggered_rules → ["red:cant_afford_from_any_angle"]
# result.explanation → "Income can't handle it, savings barely cover..."
```

**Batch (training data generation):**
```python
df["label"] = df.apply(engine.decide_row, axis=1)
```

---

## Layer 2: Downgrade Engine (`downgrade_engine.py`)

### Purpose

Layer 2 takes the Layer 1 financial label and can **only downgrade** it by at most one step
based on product quality and review reliability concerns:

```
GREEN  → YELLOW  (product/review risk adds caution)
YELLOW → RED     (product/review risk escalates concern)
RED    → RED     (already maximum risk, no change)
```

Layer 2 **cannot upgrade** a label. A RED from Layer 1 stays RED regardless of how good the
product's reviews are.

### Downgrade trigger condition

A downgrade requires **BOTH**:
- At least one **product rule** triggers (product metadata is concerning), AND
- At least one **review rule** triggers (review data is concerning).

This follows the same cross-group principle as Layer 1: a single source of concern (e.g., only
product metadata, or only reviews) is not enough to change the label. Both must agree.

### Product rules (PR1–PR3)

Use 7 product features computed from product metadata (ratings, review counts, price positioning):

| Feature | What it measures |
|---------|-----------------|
| `value_density` | Rating quality relative to price |
| `review_confidence` | Statistical confidence based on review count |
| `rating_polarization` | How split the ratings are (love-it-or-hate-it products) |
| `quality_risk_score` | Combined risk from low ratings and high variance |
| `cold_start_flag` | 1 if the product has very few reviews |
| `price_category_rank` | How expensive this product is within its category (0–1) |
| `category_rating_deviation` | How far this product's rating is from its category average |

| Rule | Conditions | Groups | What it catches |
|------|-----------|--------|----------------|
| **PR1** | `category_rating_deviation < −0.5` AND `review_confidence < 0.3` | A + B | Below-category-average rating with insufficient review evidence. |
| **PR2** | `category_rating_deviation < −0.8` AND `price_category_rank > 0.7` | A + D | Among the worst-rated products in its category while priced as premium. |
| **PR3** | `rating_polarization > 0.6` AND `cold_start_flag == 1` AND `price_category_rank > 0.5` | C + B + D | Highly polarized early reviews on a relatively expensive, new product. |

### Review rules (RR1–RR3)

Use 6 review features computed from review text, ratings, and reviewer metadata:

| Feature | What it measures |
|---------|-----------------|
| `verified_purchase_ratio` | % of reviews from verified purchasers |
| `helpful_concentration` | Whether helpfulness votes are concentrated on few reviews |
| `sentiment_spread` | Average sentiment direction (-1 to +1) |
| `review_depth_score` | How detailed/substantive the reviews are |
| `reviewer_diversity` | How many unique reviewers contributed |
| `extreme_rating_ratio` | % of reviews that are 1-star or 5-star |

| Rule | Conditions | Groups | What it catches |
|------|-----------|--------|----------------|
| **RR1** | `verified_purchase_ratio < 0.3` AND `extreme_rating_ratio > 0.8` AND `review_depth_score < 0.2` | E + J + H | Fake review pattern: shallow, extreme reviews with very few verified purchases. |
| **RR2** | `verified_purchase_ratio < 0.4` AND `helpful_concentration > 0.7` AND `reviewer_diversity < 0.5` | E + F + I | Helpfulness votes concentrated on a small, unverified reviewer set. |
| **RR3** | `sentiment_spread < −0.3` AND `review_depth_score < 0.3` AND `verified_purchase_ratio < 0.5` | G + H + E | Mostly negative, shallow reviews from largely unverified purchasers. |

### Impact on training data

In the current dataset (50K scenarios), only **224 rows** (0.4%) were downgraded by Layer 2.
This is expected — product/review concerns are relatively rare compared to financial concerns.
The `downgraded` column in `training_scenarios.csv` flags these rows.

### API

```python
from deterministic_engine.downgrade_engine import DowngradeEngine

engine = DowngradeEngine()
result = engine.evaluate(
    financial_label="GREEN",
    product_features=product_features_obj,
    review_features=review_features_obj,
)
# result.final_label → "YELLOW"  (if both product + review rules fired)
# result.was_downgraded → True
# result.product_triggers → ["PR1:underrated_unverified"]
# result.review_triggers → ["RR1:fake_review_pattern"]
```

---

## End-to-end example

```
User: monthly_income=$3,000, DI=-$800, liquid_savings=$5,700

Session (graduated):
┌─────────────────────────────────────────────────────────────────────┐
│ Tier     │ Product  │ DI after │ Savings after │ Layer 1 │ Layer 2 │
├──────────┼──────────┼──────────┼───────────────┼─────────┼─────────┤
│ budget   │ $150     │ -$800    │ $5,550        │ GREEN   │ GREEN   │
│ mid      │ $800     │ -$800    │ $4,750        │ GREEN   │ YELLOW  │ ← downgraded
│ premium  │ $2,000   │ -$800    │ $2,750        │ RED     │ RED     │ ← stopped
└─────────────────────────────────────────────────────────────────────┘

DI is negative, so all purchases come from savings.
- Budget $150: savings $5,700 → $5,550. SPR = 37 (healthy). GREEN.
- Mid $800: savings $5,550 → $4,750. SPR = 5.9. GREEN from Layer 1, but product has
  polarized reviews + fake review pattern → Layer 2 downgrades to YELLOW.
- Premium $2,000: savings $4,750 → $2,750. SPR = 1.4 (below 1.5). RED Rule 1 fires.
  Session stops.

Output: 3 rows in training_scenarios.csv, grouped under one session_id.
```
