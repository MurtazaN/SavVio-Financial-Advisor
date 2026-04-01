# How `liquid_savings` Is Calculated (Income-Based)

This document explains how **liquid_savings** is derived in the SavVio pipeline. The value is **not** raw total savings; it is an **income-based, SCF-inspired** estimate of how much of a user’s wealth is realistically available as liquid cash.

**Where it’s computed:**  
`data_pipeline/dags/src/features/financial_features.py` → `calculate_liquid_savings()`

**Where it’s used:**  
- Model pipeline: `financial_features.py`, `training_data_generator.py`, and the deterministic engine (savings_to_price_ratio, residual_utility_score, emergency_fund_months, etc.).
- Database: `financial_featured.csv` and the `financial_profiles` table include a `liquid_savings` column.

---

## Why not use `savings_balance` directly?

- **savings_balance** = total reported wealth (e.g. from surveys or synthetic data) and can include retirement accounts, illiquid assets, or inflated values.
- **liquid_savings** = amount assumed to be **actually accessible** for day-to-day spending or emergencies (checking, savings accounts, cash).

Using total balance as “liquid” would overstate how much users can spend without penalty or delay. The pipeline therefore derives a **realistic liquid portion** from total savings, with limits that depend on **income**.

---

## Income-based liquidity tiers (SCF-inspired)

The logic uses **monthly income** to assign each user to one of several **liquidity tiers**. Each tier defines:

1. **Fraction of savings that is liquid** — a percentage range of `savings_balance`.
2. **Absolute cap** — a dollar range so that, for example, a low-income user cannot have an unrealistically large “liquid” amount.

Tiers are based on patterns from the **Survey of Consumer Finances (SCF)** and similar sources (e.g. average savings by income band). References are noted in the code (e.g. US News, Yahoo Finance on average U.S. savings).

### Tier definitions

| Monthly income (upper bound) | Liquid % of savings (low–high) | Absolute cap range ($) |
|-----------------------------|--------------------------------|-------------------------|
| &lt; $1,500                  | 60% – 80%                      | $500 – $3,000           |
| $1,500 – $3,000             | 30% – 50%                      | $2,000 – $10,000        |
| $3,000 – $5,000             | 15% – 25%                      | $5,000 – $30,000        |
| $5,000 – $8,000             | 8% – 15%                       | $10,000 – $60,000       |
| $8,000+                     | 5% – 10%                       | $25,000 – $150,000      |

- Income is **monthly**.
- The “upper bound” is exclusive (e.g. &lt; $1,500 means the first tier).
- The last row is “$8,000 and above.”

---

## Two-step calculation per user

For each user:

1. **Liquid fraction**  
   - Take the tier for that user’s `monthly_income`.  
   - Draw a fraction uniformly at random between the tier’s low and high percentage.  
   - Compute:  
     `fractional_liquid = savings_balance × that_fraction`

2. **Absolute cap**  
   - Draw a cap uniformly at random between the tier’s cap_low and cap_high.  
   - Then:  
     `liquid_savings = min(fractional_liquid, cap)`

So:

- **liquid_savings** is always a **percentage of savings_balance** (step 1), but  
- **capped in dollars** so it doesn’t exceed a plausible liquid amount for that income (step 2).

If `savings_balance <= 0`, `liquid_savings` remains 0 (no liquid amount is computed for that user).

### Why different users get different values (e.g. $2,980 vs $1,876)

The **cap is drawn at random per user** in the tier's range. When the fractional liquid would be above that cap, we use the cap, so `liquid_savings` equals that random value — hence one user may get $2,980, another $1,876. When the fractional amount is below the cap, `liquid_savings` is just the fractional amount (and can be below the tier's cap_low).

---

## Examples (conceptual)

| Monthly income | savings_balance | Tier        | Example liquid % | Example cap | liquid_savings (example) |
|----------------|-----------------|------------|------------------|-------------|---------------------------|
| $1,200         | $20,000         | &lt; $1.5K | 70%              | $2,000      | min(14,000, 2,000) = **$2,000** |
| $4,000         | $50,000         | $3K–$5K   | 20%              | $18,000     | min(10,000, 18,000) = **$10,000** |
| $10,000        | $500,000        | $8K+      | 7%               | $80,000     | min(35,000, 80,000) = **$35,000** |

So:

- Low income + high balance → fraction could be large, but the **cap** keeps liquid_savings in a plausible range (e.g. $500–$3K for &lt; $1.5K/mo).
- High income + high balance → lower fraction of savings treated as liquid, with a higher cap (e.g. $25K–$150K).

---

## Reproducibility

The function uses a fixed **random_state** (default 42) so that, for the same input DataFrame and seed, **liquid_savings** is reproducible across runs.

---

## Downstream use

- **Decision engine (Layer 1):**  
  Uses `liquid_savings` (and other inputs) to compute **savings_to_price_ratio**, **residual_utility_score**, **emergency_fund_months**, **net_worth_indicator**, and thus GREEN/YELLOW/RED.

- **Training data generator:**  
  Uses `liquid_savings` and **discretionary_income** with a **DI-first** spending model: purchases consume discretionary income first; only the shortfall is taken from liquid_savings. The CSV written for training includes the (possibly session-adjusted) `liquid_savings` per scenario.

- **Inference:**  
  The same `liquid_savings` (from DB/featured CSV) is used when computing affordability for a single user–product pair.

In short: **liquid_savings** is an income-based, SCF-inspired estimate of **how much of a user’s savings is realistically liquid**, and it drives both the deterministic rules and the training/inference features.
