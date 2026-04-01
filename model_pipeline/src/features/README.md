# Feature Computation & Training Data Generation

This document covers two modules that work together to produce the training dataset:

1. **`financial_features.py`** — Computes 6 financial features for a single user-product pair (used at inference time).
2. **`training_data_generator.py`** — Generates 50K+ labeled scenarios by pairing real users with real products across graduated price tiers (used for ML training).

Source files:
- `model_pipeline/src/features/financial_features.py`
- `model_pipeline/src/features/training_data_generator.py`

---

## Part 1: Financial Features (`financial_features.py`)

### Purpose

Computes 6 derived financial features that capture how well a user can absorb a specific purchase.
These features are the inputs to the deterministic decision engine (Layer 1).

This module is used at **inference time** — when a real user queries whether they should buy a
specific product. For batch training data generation, `training_data_generator.py` reimplements
the same math in a vectorized form for performance.

### The 6 computed features

| # | Feature | Formula | What it measures |
|---|---------|---------|-----------------|
| 1 | `affordability_score` | `discretionary_income − product_price` | Can the user's monthly cash flow absorb this purchase? Negative = income alone can't cover it. |
| 2 | `price_to_income_ratio` (PIR) | `product_price / monthly_income` | How big is this purchase relative to income? PIR > 0.10 flags non-trivial purchases in RED/YELLOW rules. |
| 3 | `residual_utility_score` (RUS) | `(liquid_savings − product_price) / total_obligations` | After buying this from savings, how many months of financial runway remain? |
| 4 | `savings_to_price_ratio` (SPR) | `liquid_savings / product_price` | How many times over can savings cover this product? SPR < 1.5 means savings barely cover it. |
| 5 | `net_worth_indicator` (NWI) | `(liquid_savings − loan_amount) / monthly_income` | Is the user underwater? Negative NWI means outstanding debt exceeds savings. |
| 6 | `credit_risk_indicator` (CRI) | `(credit_score − 299) / 550` | Normalized credit score (0–1 range). A brand-new credit holder (score=300) gets CRI ≈ 0.002, not zero. |

### Input features (from database)

These are pre-computed by the data pipeline and stored in PostgreSQL:

| Feature | Source |
|---------|--------|
| `monthly_income` | User's monthly income |
| `discretionary_income` | Income minus all recurring expenses (rent, utilities, subscriptions, etc.) |
| `liquid_savings` | Accessible savings (checking + savings accounts, not retirement/locked funds) |
| `monthly_expenses` | Total monthly expenses |
| `monthly_emi` | Monthly loan EMI payments |
| `loan_amount` | Outstanding loan balance |
| `credit_score` | Credit score (300–850 range) |
| `debt_to_income_ratio` | Total debt payments / income |
| `saving_to_income_ratio` | Savings / income |
| `monthly_expense_burden_ratio` | (Expenses + EMI) / income |
| `emergency_fund_months` | Savings / (expenses + EMI) — months of runway without income |

### Cumulative spending — DI-first priority model

When a user evaluates multiple products in a session, prior purchases reduce their available
resources. The spending follows a **DI-first** priority:

1. **Discretionary income (DI) absorbs the purchase first.** DI represents the user's monthly
   cash flow after recurring expenses. If DI can cover the product price, savings are untouched.

2. **If DI is insufficient, liquid savings cover the shortfall.** Only the gap between the
   product price and available DI is deducted from savings.

**Example — user with DI=$1,200 and savings=$8,000 buying 3 products:**

| Purchase | Price | DI covers | Savings cover | DI after | Savings after |
|----------|------:|----------:|--------------:|---------:|--------------:|
| Budget   | $200  | $200      | $0            | $1,000   | $8,000        |
| Mid      | $800  | $800      | $0            | $200     | $8,000        |
| Premium  | $1,500| $200      | $1,300        | $0       | $6,700        |

DI absorbed the first two purchases entirely. On the third, DI only had $200 left, so savings
covered the remaining $1,300.

**Why not deduct from both equally?** Because that double-counts the spending. A user with
$1,200 DI buying a $200 product still has $1,000 DI and the same savings. Their savings
shouldn't decrease for a purchase their income can handle.

### API

```python
from features.financial_features import compute_affordability

result = compute_affordability(
    user_financial_profile={
        "monthly_income": 5000.0,
        "discretionary_income": 1200.0,
        "liquid_savings": 8000.0,
        "monthly_expenses": 2800.0,
        "monthly_emi": 1000.0,
        "loan_amount": 15000.0,
        "credit_score": 720,
    },
    product_price=799.99,
    cumulative_spend=500.0,  # user already bought $500 worth this session
)
# result.affordability_score → -99.99
# result.savings_to_price_ratio → 10.0
```

---

## Part 2: Training Data Generator (`training_data_generator.py`)

### Purpose

Generates the labeled training dataset by pairing real user financial profiles with real
products, computing features, and applying the deterministic engine to produce
GREEN/YELLOW/RED labels. The output CSV is used to train the downstream ML model.

### Sampling strategies

The generator supports three strategies (controlled by `graduated` and `stratified` parameters):

#### 1. Graduated (default, recommended)

Each user evaluates one product per price tier in ascending order: **budget → mid → premium**.
This simulates a realistic shopping session where a user considers increasingly expensive items.

**Key behaviors:**
- **Cumulative spending** — After each GREEN/YELLOW purchase, the product price is deducted
  from the user's available resources (DI first, then savings) before evaluating the next tier.
- **RED early-stop** — If a user gets RED on any tier, they stop. No further tiers are evaluated
  (the engine already determined this user can't afford more).
- **Session grouping** — Each user's graduated journey is assigned a `session_id` and rows are
  sorted so tiers appear consecutively: budget → mid → premium.

**Price tier bins (current):**

| Tier    | Price range      | Product pool |
|---------|-----------------|-------------|
| budget  | $100 – $500     | 6,700       |
| mid     | $500 – $1,500   | 881         |
| premium | $1,500+         | 454         |

Products under $100 are excluded — they're always GREEN and provide no learning signal.

#### 2. Stratified

Equal representation across income × price bracket combinations (3 income × 3 price = 9 cells).
Each cell gets `n_scenarios / 9` rows. Used when you want balanced coverage without the
session/cumulative-spending semantics.

#### 3. Random

Pure uniform random pairing. Legacy mode for quick experiments.

### Output schema

Each row in `training_scenarios.csv` represents one (user, product) evaluation:

| Column | Description |
|--------|-------------|
| `user_id` | User identifier |
| `session_id` | Groups rows belonging to the same user's graduated journey (0, 1, 2, ...) |
| `price_tier` | Which tier this product was sampled from: `budget`, `mid`, or `premium` |
| `product_id` | Product identifier |
| `product_price` | Product price |
| `monthly_income` | User's monthly income (unchanged across tiers) |
| `discretionary_income` | User's DI **at the time of this evaluation** (decreases across tiers as prior purchases consume it) |
| `liquid_savings` | User's savings **at the time of this evaluation** (decreases only when DI is exhausted) |
| `affordability_score` | Computed feature (6 total) |
| `price_to_income_ratio` | Computed feature |
| `residual_utility_score` | Computed feature |
| `savings_to_price_ratio` | Computed feature |
| `net_worth_indicator` | Computed feature |
| `credit_risk_indicator` | Computed feature |
| `financial_label` | Final label: GREEN / YELLOW / RED (after both Layer 1 and Layer 2) |
| `downgraded` | `1` if Layer 2 (downgrade engine) changed the label, `0` otherwise |

### How a graduated session works — step by step

```
User U31070: DI = -$798.53, Savings = $16,354.36

1. Budget tier ($189.99):
   - DI is negative → savings cover $189.99
   - Savings: $16,354 → $16,164
   - Engine evaluates: GREEN
   - User proceeds to next tier.

2. Mid tier ($1,299.95):
   - DI still negative → savings cover $1,299.95
   - Savings: $16,164 → $14,864
   - Engine evaluates: GREEN
   - User proceeds to next tier.

3. Premium tier ($1,889.00):
   - DI still negative → savings cover $1,889
   - Savings: $14,864 → $12,975
   - Engine evaluates: RED (savings_to_price_ratio dropped below threshold)
   - Session stops. User gets 3 rows in the output.
```

### Layer 2 downgrade

After Layer 1 (financial engine) labels all scenarios, Layer 2 (downgrade engine) optionally
downgrades labels based on product/review quality. The `downgraded` column flags which rows
were affected. See `decision_engine.md` for Layer 2 rule details.

### Truncation and ordering

When the generator produces more rows than `n_scenarios`, it drops complete sessions from the
end (never splits a session mid-way). Session IDs are renumbered to be consecutive (0, 1, 2, ...),
and rows are sorted by `(session_id, tier_order)` so each user's journey is contiguous.

### API

```python
from features.training_data_generator import generate_scenarios

scenarios = generate_scenarios(
    financial_profiles=financial_df,   # DataFrame from DB
    products=products_df,              # DataFrame from DB
    reviews_df=reviews_df,             # Optional — enables Layer 2
    n_scenarios=50000,
    random_state=42,
    graduated=True,                    # default
)
scenarios.to_csv("training_scenarios.csv", index=False)
```

### Current label distribution (50K scenarios)

| Label  | Count  | %     |
|--------|-------:|------:|
| GREEN  | 36,391 | 72.8% |
| YELLOW | 5,176  | 10.4% |
| RED    | 8,432  | 16.9% |

224 rows were downgraded by Layer 2.
## Product & Review Features (Layer 2 Inputs)

This document explains the **product** and **review** feature engineering used by the Layer 2 downgrade engine.

Source files:
- `model_pipeline/src/features/product_features.py`
- `model_pipeline/src/features/review_features.py`

These features are **not** used by the Layer 1 financial engine. They are used by:
- `model_pipeline/src/deterministic_engine/downgrade_engine.py` (Layer 2), which can **only downgrade** the financial label by one step when both product and review risk corroborate.

---

## Why these features exist

Layer 1 answers: **“Can the user afford it?”**

Layer 2 adds a separate question: **“Even if they can afford it, is this product/review pattern risky enough that we should add caution?”**

Downgrading requires:
- ≥ 1 product-rule trigger **and**
- ≥ 1 review-rule trigger

This “two independent sources” requirement reduces false positives from any single weak signal.

---

## Product features (`product_features.py`)

### Inputs required per product row

Batch computation requires these columns:
- `price`
- `average_rating`
- `rating_number`
- `rating_variance`

For category-normalized features, it also expects:
- `category` (if missing, the code falls back to treating each product as its own category)

### Output features (7)

The batch API `compute_product_features_batch(products_df)` returns a DataFrame with these columns added:

1. **`value_density` (VD)**  
   - **Formula**: `average_rating / log(price + 1)`  
   - **Intuition**: a “value per dollar” proxy. A high rating at a very high price is less impressive than the same rating at a low price.

2. **`review_confidence` (RC)**  
   - **Formula**: `log(rating_number + 1) / log(max_rating_number + 1)`  
   - **Range**: ~0 to 1  
   - **Intuition**: how much evidence backs the rating, normalized to the largest review count in the dataset.

3. **`rating_polarization` (RP)**  
   - **Formula**: `rating_variance / (average_rating * (5 - average_rating) + EPS)`  
   - **Intuition**: captures “love it or hate it” rating patterns. Normalizes variance by the maximum possible variance at the given mean rating.

4. **`quality_risk_score` (QRS)**  
   - **Formula**: `(5 - average_rating) * (1 - review_confidence)`  
   - **Intuition**: low ratings are more risky when evidence is weak. This is high when `average_rating` is low and `rating_number` is small.

5. **`cold_start_flag` (CSF)**  
   - **Formula**: `1 if rating_number < 10 else 0`  
   - **Intuition**: products with very few reviews are inherently less reliable.

6. **`price_category_rank` (PCR)**  
   - **Formula**: `(price - category_min_price) / (category_max_price - category_min_price + EPS)`  
   - **Range**: ~0 to 1  
   - **Intuition**: how expensive this product is **within its category**.

7. **`category_rating_deviation` (CRD)**  
   - **Formula**: `average_rating - category_mean_rating`  
   - **Intuition**: how far above/below category average this product is rated.

### Category stats precomputation

`compute_category_stats(products_df)` builds a mapping per `category`:
- `min_price`
- `max_price`
- `mean_rating`

Those stats are used for PCR and CRD.

### APIs

**Single product row**

```python
from features.product_features import compute_category_stats, compute_product_features

category_stats = compute_category_stats(products_df)
max_rating_number = float(products_df["rating_number"].max() or 0.0)
pf = compute_product_features(product_row, category_stats, max_rating_number)
```

**Batch**

```python
from features.product_features import compute_product_features_batch

products_with_features = compute_product_features_batch(products_df)
```

---

## Review features (`review_features.py`)

### Inputs expected per review row

`compute_review_features_batch(reviews_df)` groups by `product_id` and expects:
- `product_id` (required)

It also uses these if present (with safe defaults if missing):
- `verified_purchase` (defaults to False)
- `helpful_vote` (defaults to 0)
- `rating` (defaults to 0)
- `review_text` (defaults to empty string)
- `user_id` (defaults to None)

### Output features (6)

The batch API returns a DataFrame indexed by `product_id` with these columns:

1. **`verified_purchase_ratio` (VPR)**  
   - **Formula**: `(# verified_purchase == True) / n_reviews`  
   - **Intuition**: low VPR can indicate unreliable reviews.

2. **`helpful_concentration` (HC)**  
   - **Formula**: `max(helpful_vote) / (sum(helpful_vote) + EPS)`  
   - **Intuition**: when one review dominates helpfulness votes, the overall review set may be less representative.

3. **`sentiment_spread` (SS)**  
   - **Formula**: `(# ratings >= 4 - # ratings <= 2) / n_reviews`  
   - **Range**: -1 to 1  
   - **Intuition**: direction of sentiment (positive vs negative) without using NLP.

4. **`review_depth_score` (RDS)**  
   - **Formula**: `min(mean_word_count / 100, 1)`  
   - **Range**: 0 to 1  
   - **Intuition**: shallow reviews are less informative; detailed reviews score higher.

5. **`reviewer_diversity` (RD)**  
   - **Formula**: `n_unique_reviewers / n_reviews`  
   - **Range**: 0 to 1  
   - **Intuition**: a product whose reviews come from many unique users is more reliable than one dominated by a few accounts.

6. **`extreme_rating_ratio` (ERR)**  
   - **Formula**: `(# rating == 1 or rating == 5) / n_reviews`  
   - **Intuition**: unusually high extremes can indicate manipulation or polarization.

### Empty-review behavior

If a product has **no reviews**, the code returns **neutral/zeroed** features:
- VPR=0, HC=0, SS=0, RDS=0, RD=0, ERR=0

This ensures downstream code can still run and makes the downgrade engine conservative (it typically needs both product+review triggers).

### APIs

**Single product’s reviews**

```python
from features.review_features import compute_review_features

rf = compute_review_features(reviews_for_product_df)
```

**Batch**

```python
from features.review_features import compute_review_features_batch

review_features_by_product = compute_review_features_batch(reviews_df)
# index: product_id
```

---

## How these features connect to Layer 2 rules

The downgrade engine (`downgrade_engine.py`) consumes:
- Product features: `value_density`, `review_confidence`, `rating_polarization`, `quality_risk_score`, `cold_start_flag`, `price_category_rank`, `category_rating_deviation`
- Review features: `verified_purchase_ratio`, `helpful_concentration`, `sentiment_spread`, `review_depth_score`, `reviewer_diversity`, `extreme_rating_ratio`

And applies rule triggers such as:
- PR1: below-category-average rating AND weak review evidence (CRD + RC)
- RR1: fake-review pattern (VPR + ERR + RDS)

Only when **at least one PR and at least one RR** trigger will the label be downgraded by one step.

# Price Bin Analysis — Product Price Tiers for Training Data

This document tracks the evolution of product price bins used in `training_data_generator.py`
to create balanced training data for the SavVio ML model. Each configuration was tested against
real user financial profiles and product data to measure label distribution (GREEN / YELLOW / RED).

---

## Why price bins matter

The deterministic engine evaluates whether a user should buy a product. If most products in the
training data are cheap (under $100), almost every scenario gets labeled GREEN because the purchase
is trivially affordable. The model never learns to distinguish YELLOW or RED situations.

Price bins control how products are sampled during graduated session generation. Each user evaluates
one product per tier (budget → mid → premium), with cumulative spending carrying forward.

**Key user financial statistics (from real data):**

| Stat | Monthly Income | Discretionary Income | Liquid Savings |
|------|---------------:|---------------------:|---------------:|
| P10  | $1,454         | -$8,238              | $2,304         |
| P25  | $2,658         | -$2,009              | $5,660         |
| P50  | $3,998         | $721                 | $11,811        |
| P75  | $5,352         | $1,728               | $23,103        |
| P90  | $6,521         | $2,734               | $33,562        |

33.1% of users have **negative** discretionary income — they're already spending more than they earn
monthly and rely entirely on savings for purchases.

**Product price distribution:**

| Stat  | Price    |
|-------|----------|
| P50   | $26.99   |
| P75   | $33.62   |
| P90   | $87.95   |
| P95   | $169.17  |
| P99   | $805.73  |

91.5% of products are under $100. The median price is $26.99. This extreme left-skew means
naive random sampling produces almost entirely GREEN labels.

---

## Configurations tested

### Config 1: Original bins `[0, 50, 200, ∞]` — 3 tiers

| Tier    | Range        | Products |
|---------|-------------|----------|
| budget  | $0 – $50    | 88,412   |
| mid     | $50 – $200  | 4,915    |
| premium | $200+       | 1,000    |

**Result:** GREEN 91.7% / YELLOW 2.0% / RED 6.4%

**Problem:** Budget tier is dominated by cheap products. The $6.99 and $26.99 mass points
mean budget purchases are trivially GREEN for everyone. Model barely sees YELLOW.

---

### Config 2: 4-tier bins `[0, 100, 500, 1500, ∞]`

| Tier    | Range            | Products |
|---------|-----------------|----------|
| budget  | $0 – $100       | 86,292   |
| mid     | $100 – $500     | 6,700    |
| high    | $500 – $1,500   | 881      |
| premium | $1,500+         | 454      |

**Result:** GREEN 79.6% / YELLOW 7.5% / RED 12.9%

**Improvement:** Adding the $500+ tiers forced the engine to evaluate purchases that genuinely
stress users' finances. But the budget tier ($0–$100) still contributed only GREEN labels — it
was wasted signal.

---

### Config 3: `[100, 500, 1500, ∞]` — 3 tiers, no sub-$100 (current)

Eliminated products under $100 entirely. A $6.99 product is always GREEN — no learning signal.

| Tier    | Range            | Products | % of pool |
|---------|-----------------|----------|-----------|
| budget  | $100 – $500     | 6,700    | 83.4%     |
| mid     | $500 – $1,500   | 881      | 11.0%     |
| premium | $1,500+         | 454      | 5.6%     |

**Result:** GREEN 72.8% / YELLOW 10.4% / RED 16.9%

**Why this wins:**
- Best label balance of all tested configurations.
- Every product in the pool actually challenges users' finances.
- Bin boundaries align with engine PIR thresholds:
  - $100–$500 = 10–15% of a low-income user's paycheck → triggers YELLOW rules.
  - $500–$1,500 = 10–20% of median income → triggers YELLOW/RED rules.
  - $1,500+ = significant even for high earners → triggers RED rules.

---

### Other configurations tested

| Config | Bins | GREEN | YELLOW | RED |
|--------|------|------:|-------:|----:|
| `[100, 400, 1000, ∞]` — 3 tiers | $100/$400/$1K+ | 77.3% | 7.7% | 15.0% |
| `[100, 300, 800, 2000, ∞]` — 4 tiers | $100/$300/$800/$2K+ | 75.5% | 10.5% | 14.0% |
| `[0, 15, 50, ∞]` — 3 tiers (early) | $0/$15/$50+ | 15%/68%/17% (bins) | — | — |

Config `[100, 400, 1000+]` was considered but produced fewer YELLOW labels (7.7% vs 10.4%).
Config `[100, 300, 800, 2000+]` with 4 tiers had comparable balance but thinner product pools
in the upper tiers (314 premium products vs 454 in the chosen config).

---

## Final configuration

```python
PRICE_BINS  = [100, 500, 1_500, float("inf")]
PRICE_LABELS = ["budget", "mid", "premium"]

INCOME_BINS  = [0, 3_000, 5_000, float("inf")]
INCOME_LABELS = ["low", "mid", "high"]
```

Label distribution on 50,000 scenarios: **GREEN 72.8% / YELLOW 10.4% / RED 16.9%**

---

## Cross-dataset price analysis

For reference, we also analyzed price distributions across Amazon product categories to
understand which datasets produce the best bin balance:

| Dataset | Best 3-tier bins | Split |
|---------|-----------------|-------|
| Electronics (1.6M items) | `[0, 15, 50, ∞]` | 35% / 37% / 28% |
| Video Games (62K items) | `[0, 15, 40, ∞]` | 31% / 41% / 29% |
| General product_featured (94K items) | `[0, 15, 50, ∞]` | 15% / 68% / 17% |

The general product dataset has a 39% mass point at $26.99, making balanced low-range bins
impossible. Our solution: skip the low range entirely and use bins starting at $100, which
aligns with the financial significance threshold for our user base.
