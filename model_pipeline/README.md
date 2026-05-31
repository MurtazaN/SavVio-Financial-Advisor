# SavVio — Model Development Pipeline

**Team:** Murtaza Nipplewala, Niraj Mehta, Wen-Hsin Su, Pranathi Bombay, Rishabh Joshi, Sanjana Patnam

---

## ML Pipeline Structure

```
SavVio/
└── model_pipeline/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── models/                         # Local dev model storage (gitignored)
    │   ├── checkpoints/
    │   ├── artifacts/
    │   └── preprocessing/
    ├── src/
    │   ├── run_pipeline.py             # End-to-end ML pipeline entrypoint
    │   ├── config.py                   # Centralized configuration
    │   ├── push_to_registry.py         # GCP Artifact Registry push script
    │   ├── data/
    │   │   ├── db_loader.py            # Reads from PostgreSQL
    │   │   └── validate_data.py        # Schema validation
    │   ├── features/
    │   │   ├── feature_engineering.py   # Orchestrator: build_training_data()
    │   │   ├── affordability.py         # 6 computed financial features (shared)
    │   │   ├── financial_features.py    # Inference-time affordability wrapper
    │   │   ├── product_features.py      # 7 product quality features (Layer 2)
    │   │   ├── review_features.py       # 6 review reliability features (Layer 2)
    │   │   ├── feature_preprocessing.py # FeaturePipeline: impute, encode, scale, drop
    │   │   └── training_data_generator.py  # Stratified scenario sampling
    │   ├── deterministic_engine/
    │   │   ├── financial_engine.py      # Layer 1: financial GREEN/YELLOW/RED rules
    │   │   ├── downgrade_engine.py      # Layer 2: product/review downgrade rules
    │   │   └── labeling_pipeline.py     # Orchestrates Layer 1 + Layer 2 labeling
    │   ├── core_models/
    │   │   ├── train.py                 # XGBoost, LightGBM, XGB-Linear
    │   │   ├── evaluate.py              # Metrics, visualizations, MLflow logging
    │   │   ├── optuna_tuner.py          # Bayesian hyperparameter optimization
    │   │   └── sensitivity_analysis.py  # Optuna-based param importance analysis
    │   ├── guards/
    │   │   └── bias_detection.py        # Fairlearn demographic parity + equalized odds
    │   └── llm/
    │       ├── __init__.py              # Package exports
    │       ├── config.py               # LLM configuration (provider, keys, thresholds)
    │       ├── llm_provider.py         # Provider abstraction (Mock / OpenAI / Gemini / Claude)
    │       ├── intent_parser.py        # Role 1: NLU — intent detection + product extraction
    │       ├── product_resolver.py     # pgvector similarity search for product resolution
    │       ├── response_generator.py   # Role 2: conversational recommendation generation
    │       ├── guardrails.py           # Output safety verification (6 code-level checks)
    │       ├── prompt_engin.py         # Legacy interface (backward compatible facade)
    │       └── prompts/
    │           ├── system_prompt.py     # SavVio persona + critical response rules (v1.0)
    │           ├── intent_prompt.py     # Intent extraction prompt template (v1.0)
    │           └── response_templates.py # GREEN/YELLOW/RED templates + rule explanations
    └── tests/
```

---

## Quick Reference: Tools by Phase

## CI/CD Deployment Note

- Workflow: `.github/workflows/modelpipeline_ci.yml`
- Build source: `model_pipeline/docker-compose.yml` (`ml-trainer` service via `docker buildx bake`)
- Image target: `us-east1-docker.pkg.dev/<project>/savvio-dev-docker-repo/savvio-model-pipeline`
- Deploy target: Terraform-managed Cloud Run service `savvio-ai`
- Infra source of truth: `.github/workflows/terraform.yml` + `deployment_pipeline/terraform/environments/dev/*`

| Phase | Primary Tools | Alternatives | CI/CD Gate |
|-------|--------------|--------------|------------|
| Data Loading | DVC, GCS, Pandas | Polars, LakeFS | Data version + schema check |
| Feature Engineering | Pandas, NumPy, scikit-learn | Polars, Feature-engine | Feature schema validation |
| Deterministic Engine | Pure Python (2 layers) | — | Unit tests must pass |
| Model Training | XGBoost, LightGBM, scikit-learn | — | Reproducible training run |
| Hyperparameter Tuning | Optuna | RandomizedSearchCV | Best-run tracking required |
| Validation & Metrics | sklearn.metrics, Matplotlib | Seaborn, Plotly | Minimum F1 / AUC threshold |
| Bias Detection | Fairlearn | AIF360, TFMA | Block on severe disparity |
| Bias Mitigation | Fairlearn, imbalanced-learn | scikit-learn threshold | Re-evaluate until gates pass |
| Model Selection | MLflow UI | Custom dashboards | Bias mitigation gate must pass |
| Sensitivity & Explainability | Optuna importance, Matplotlib | SHAP, LIME (planned) | Stability report required |
| Experiment Tracking | MLflow | Weights & Biases | Run metadata completeness |
| Model Registry Push | GCP Artifact Registry, Vertex AI | MLflow Registry | Push only on all gates pass |
| CI/CD Automation | GitHub Actions, Docker | Cloud Build, Jenkins | src ↔ test ↔ DB ↔ ML pipeline |
| LLM Integration | OpenRouter (Gemini 2.0), sentence-transformers, pgvector | OpenAI, Claude | Guardrail checks (G1-G6) must pass |
| Monitoring & Dashboard | Evidently, Arize | WhyLabs, GCP Monitoring | Drift + latency alerts |

---

## Model Pipeline Execution Order

```
1.  Load Data (PostgreSQL via data pipeline)
        ↓
2.  Feature Engineering + Scenario Generation + Layer 1/Layer 2 Deterministic Labels
        ↓
3.  3-Way Stratified Split (train 60% / val 20% / test 20%)
        ↓
4.  Baseline Training (XGBoost, LightGBM, XGB-Linear)
        ↓
5.  Hyperparameter Tuning (Optuna on best baseline)
        ↓
6.  Validation on Validation Set (per-candidate metrics + visualizations)
        ↓
7.  Model Selection (F1 ranking + bias gate)
        ↓
8.  Final Evaluation on Held-Out Test Set
        ↓
9.  Sensitivity Analysis (Optuna hyperparameter importance)
        ↓
10. Experiment Tracking (MLflow — all runs, artifacts, comparisons)
        ↓
11. Model Registry Push (MLflow registry + GCP Artifact Registry)
        ↓
12. CI/CD Automation (Dockerized)
        ↓
13. LLM Integration (Intent Parsing + Response Generation + Guardrails)
```

---

## Table of Contents

1. [Phase 1 — Data Loading](#phase-1--data-loading)
2. [Phase 2 — Feature Engineering](#phase-2--feature-engineering)
3. [Phase 3 — Deterministic Decision Engine](#phase-3--deterministic-decision-engine)
4. [Phase 4 — Model Training](#phase-4--model-training)
5. [Phase 5 — Hyperparameter Tuning](#phase-5--hyperparameter-tuning)
6. [Phase 6 — Validation & Metrics](#phase-6--validation--metrics)
7. [Phase 7 — Bias Detection](#phase-7--bias-detection)
8. [Phase 8 — Bias Mitigation](#phase-8--bias-mitigation)
9. [Phase 9 — Model Selection](#phase-9--model-selection)
10. [Phase 10 — Sensitivity & Explainability](#phase-10--sensitivity--explainability)
11. [Phase 11 — Experiment Tracking](#phase-11--experiment-tracking)
12. [Phase 12 — Model Registry Push](#phase-12--model-registry-push)
13. [Phase 13 — CI/CD Automation](#phase-13--cicd-automation)
14. [Phase 14 — LLM Integration](#phase-14--llm-integration)
15. [Phase 15 — Monitoring & Dashboard](#phase-15--monitoring--dashboard)
16. [Phase 16 — Testing](#phase-16--testing)
17. [Phase 17 — Operational Risks & Guardrails](#phase-17--operational-risks--guardrails)
18. [Phase 18 — Dockerize Model Development](#phase-18--dockerize-model-development)
19. [Model Candidates — Selection Rationale](#model-candidates--selection-rationale)
20. [Deliverable Checklist](#deliverable-checklist)

---

### Phase 1 — Data Loading

**Objective:** Load the latest versioned and validated datasets from the data pipeline output and ensure reproducible model inputs.

**Tasks:**
- Configure DVC remote to point to GCS bucket
- Pull versioned feature artifacts:
  ```bash
  cd data_pipeline/dags/data
  dvc pull
  ```
- Validate file existence for all three feature files
- Run schema checks (Pandera / Great Expectations) — verify column names, types, null rates
- Log DVC commit hash and GCS artifact path for reproducibility tracing
- Load financial profiles and products from PostgreSQL
- Construct Green/Yellow/Red labels from deterministic engine outputs for supervised training

**Tools:**

| Tool | Purpose |
|------|---------|
| DVC + GCS | Pull versioned artifacts |
| Pandas | Load tabular and JSONL data |
| Pandera / Great Expectations | Schema and data contract checks |

---

### Phase 2 — Feature Engineering

**Objective:** Transform raw DB tables into a model-ready feature matrix with deterministic GREEN/YELLOW/RED labels via a two-layer labeling system.

**Source Files:**
- `features/feature_engineering.py` — Orchestrator: `build_training_data()`, `generate_training_data()`, `transform_features()`
- `features/affordability.py` — Canonical 6 financial feature computation (shared by training + inference)
- `features/financial_features.py` — Inference-time single-pair wrapper returning `AffordabilityResult`
- `features/product_features.py` — 7 product quality features for Layer 2 downgrade engine
- `features/review_features.py` — 6 review reliability features for Layer 2 downgrade engine
- `features/feature_preprocessing.py` — `FeaturePipeline` class (impute → encode → scale → drop)
- `features/training_data_generator.py` — Stratified scenario sampling across income × price brackets

**Pipeline:**
1. `generate_training_data()` — Load data from PostgreSQL, sample pairs via `generate_scenarios()`, compute 6 financial features, apply Layer 1 + Layer 2 deterministic labels
2. `transform_features()` — Run the `FeaturePipeline` (impute → encode → scale → drop non-features)
3. `build_training_data()` — Orchestrator that calls 1 then 2 and returns `(X, y, scenarios_raw)`

**Feature Groups:**

| Group | Features | Source |
|-------|----------|--------|
| Financial (DB) | `discretionary_income`, `debt_to_income_ratio`, `saving_to_income_ratio`, `monthly_expense_burden_ratio`, `emergency_fund_months` | financial_profiles table |
| Product (DB) | `price`, `average_rating`, `rating_number`, `rating_variance` | products table |
| Computed Financial (6) | `affordability_score`, `price_to_income_ratio`, `residual_utility_score`, `savings_to_price_ratio`, `net_worth_indicator`, `credit_risk_indicator` | `affordability.py` |
| Computed Product (7) | `value_density`, `review_confidence`, `rating_polarization`, `quality_risk_score`, `cold_start_flag`, `price_category_rank`, `category_rating_deviation` | `product_features.py` |
| Computed Review (6) | `verified_purchase_ratio`, `helpful_concentration`, `sentiment_spread`, `review_depth_score`, `reviewer_diversity`, `extreme_rating_ratio` | `review_features.py` |
| Categorical | `employment_status`, `has_loan`, `region` | financial_profiles table |

**Scenario Generation:**
- Stratified sampling across 9 (income × price) bracket cells for balanced representation
- Income brackets: low ($0–$3K), mid ($3K–$7K), high ($7K+)
- Price brackets: budget ($100–$500), mid ($500–$1.5K), premium ($1.5K+)
- 50,000 scenarios by default (configurable via `Config.N_SCENARIOS`)
- Each scenario = one (user, product) pair with computed features + deterministic label
- Label column: `final_recommendation` (GREEN/YELLOW/RED after both Layer 1 + Layer 2)

**FeaturePipeline (`feature_preprocessing.py`):**

A unified sklearn `Pipeline` persisted as a single artifact (`feature_pipeline.pkl`):

| Step | Class | What it does |
|------|-------|--------------|
| 1. Imputer | `MissingValueImputer` | Median for financial/product numerics, 0 for `rating_variance`, median for computed features, 'Unknown' for categoricals |
| 2. Encoder | `CategoricalEncoder` | `OrdinalEncoder` on categorical columns, unknown categories → -1 |
| 3. Scaler | `NumericScaler` | `StandardScaler` on all numeric features |
| 4. Dropper | `FeatureDropper` | Drop IDs, text blobs, labels, and `product_price` before training |

**Tasks:**
- [x] Handle missing values — median imputation for financial numerics, 0 for rating_variance, 'Unknown' for categoricals
- [x] Encode categorical features via OrdinalEncoder (saved as artifact for inference)
- [x] Scale numeric features via StandardScaler (saved as artifact for inference)
- [x] Save raw scenarios as versioned CSV artifact
- [x] Compute 7 product features and 6 review features for Layer 2 downgrade
- [x] Implement stratified sampling across income × price brackets

**Tools:**

| Tool | Purpose |
|------|---------|
| Pandas / NumPy | Feature construction |
| scikit-learn | OrdinalEncoder, StandardScaler, Pipeline |
| joblib | Pipeline artifact serialization |

---

### Phase 3 — Deterministic Decision Engine

**Location:** `deterministic_engine/` — three source files:
- `financial_engine.py` — Layer 1: financial GREEN/YELLOW/RED rules
- `downgrade_engine.py` — Layer 2: product/review-based downgrade (GREEN→YELLOW, YELLOW→RED)
- `labeling_pipeline.py` — Orchestrates Layer 1 → Layer 2 for batch labeling

The deterministic engine is a **two-layer** rule-based labeling system. It is built **before** model training because it generates the labels (GREEN/YELLOW/RED) that the ML model trains on. Its output is authoritative — neither the ML model nor the LLM layer can override it.

**Architecture:**

```
User + Product
     │
     ▼
┌──────────────────────────────┐
│  Layer 1: Financial Engine   │  Uses 11 financial features (5 DB + 6 computed).
│  (financial_engine.py)       │  Evaluates 4 RED rules, then 5 YELLOW rules.
│                              │  Output: GREEN / YELLOW / RED
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Layer 2: Downgrade Engine   │  Uses 7 product + 6 review features.
│  (downgrade_engine.py)       │  Can only downgrade by 1 step (GREEN→YELLOW, YELLOW→RED).
│                              │  Requires BOTH product AND review rules to fire.
└──────────────┬───────────────┘
               │
               ▼
         Final Label
    (GREEN / YELLOW / RED)
```

---

#### Layer 1 — Financial Engine (`financial_engine.py`)

**Design Principle:** Every rule combines signals from at least TWO independent correlation groups to avoid false triggers from a single underlying cause.

**Correlation Groups:**

| Group | Features |
|-------|----------|
| Group 1 — Income capacity | `affordability_score`, `discretionary_income`, `price_to_income_ratio` |
| Group 2 — Savings depth | `saving_to_income_ratio`, `savings_to_price_ratio`, `emergency_fund_months`, `residual_utility_score` |
| Group 3 — Debt burden | `debt_to_income_ratio`, `monthly_expense_burden_ratio` |
| Group 4 — Independent | `credit_risk_indicator`, `net_worth_indicator` |

**Features Used — 11 total (5 DB + 6 computed), all financial:**
- DB: `discretionary_income`, `debt_to_income_ratio`, `saving_to_income_ratio`, `monthly_expense_burden_ratio`, `emergency_fund_months`
- Computed: `affordability_score`, `price_to_income_ratio`, `residual_utility_score`, `savings_to_price_ratio`, `net_worth_indicator`, `credit_risk_indicator`

### RED Rules (4 compound AND rules)

Each rule crosses at least 2 correlation groups and includes a `price_to_income_ratio` escape hatch so trivial purchases never trigger RED. RED returns immediately on the first rule that fires — no further evaluation.

| Rule | Groups Crossed | Condition |
|------|----------------|-----------|
| RED 1 — Can't afford from any angle | 1 + 2 | `affordability_score < 0` AND `savings_to_price_ratio < 1.5` AND `price_to_income_ratio > 0.10` |
| RED 2 — Maxed budget, significant purchase | 3 + 1 + 2 | `monthly_expense_burden_ratio > 0.80` AND `price_to_income_ratio > 0.20` AND `emergency_fund_months < 3.0` |
| RED 3 — Underwater, no surplus | 4 + 1 + 2 | `net_worth_indicator < -2.0` AND `affordability_score < 0` AND `price_to_income_ratio > 0.15` AND `emergency_fund_months < 3.0` |
| RED 4 — Paycheck-to-paycheck | 2 + 3 + 1 | `emergency_fund_months < 1.0` AND `debt_to_income_ratio > 0.30` AND `price_to_income_ratio > 0.10` |

### YELLOW Rules (5 compound AND rules, require ≥1 to trigger)

Each rule crosses at least 2 correlation groups. YELLOW triggers when **at least 1 rule fires**.

| Rule | Groups Crossed | Condition |
|------|----------------|-----------|
| YELLOW 1 — Income pressure | 1 + 2 | `affordability_score < 0` AND `price_to_income_ratio > 0.25` AND `savings_to_price_ratio < 5.0` |
| YELLOW 2 — Savings strain | 2 + 1 | `savings_to_price_ratio < 5.0` AND `price_to_income_ratio > 0.10` |
| YELLOW 3 — Debt stress | 3 + 1 + 2 | `monthly_expense_burden_ratio > 0.70` AND `price_to_income_ratio > 0.10` AND `savings_to_price_ratio < 5.0` |
| YELLOW 4 — Low resilience | 2 + 1 | `emergency_fund_months < 3.0` AND `affordability_score < 0` |
| YELLOW 5 — Weak profile | 4 + 1 + 2 | `credit_risk_indicator < 0.35` AND `net_worth_indicator < 1.0` AND `price_to_income_ratio > 0.15` AND `savings_to_price_ratio < 5.0` |

### GREEN — Default

No RED rules fired and no YELLOW rules triggered.

### Error Handling

- Missing financial fields (None) → `_safe()` raises `ValueError`
- NaN values → `_safe()` raises `ValueError`
- The engine requires all 11 features to be present — missing data must be handled upstream in feature engineering

---

#### Layer 2 — Downgrade Engine (`downgrade_engine.py`)

Layer 2 takes the Layer 1 financial label and can **only downgrade** it by at most one step based on product quality and review reliability concerns. It **cannot upgrade** a label.

**Downgrade trigger condition:** Requires **BOTH** at least one product rule trigger **AND** at least one review rule trigger.

**Product Rules (PR1–PR3)** — Use 7 product features:

| Rule | Conditions | What it catches |
|------|------------|-----------------|
| PR1 | `category_rating_deviation < −0.5` AND `review_confidence < 0.3` | Below-category-average rating with insufficient review evidence |
| PR2 | `category_rating_deviation < −0.8` AND `price_category_rank > 0.7` | Worst-rated in category while priced as premium |
| PR3 | `rating_polarization > 0.6` AND `cold_start_flag == 1` AND `price_category_rank > 0.5` | Polarized early reviews on an expensive, new product |

**Review Rules (RR1–RR3)** — Use 6 review features:

| Rule | Conditions | What it catches |
|------|------------|-----------------|
| RR1 | `verified_purchase_ratio < 0.3` AND `extreme_rating_ratio > 0.8` AND `review_depth_score < 0.2` | Fake review pattern (shallow, extreme, unverified) |
| RR2 | `verified_purchase_ratio < 0.4` AND `helpful_concentration > 0.7` AND `reviewer_diversity < 0.5` | Helpfulness concentrated on small, unverified reviewer set |
| RR3 | `sentiment_spread < −0.3` AND `review_depth_score < 0.3` AND `verified_purchase_ratio < 0.5` | Negative, shallow, unverified reviews |

**Impact:** In the current dataset (50K scenarios), ~224 rows (0.4%) are downgraded by Layer 2.

---

### Tasks

- [x] Implement Layer 1 RED rules — compound AND logic with PIR escape hatch
- [x] Implement Layer 1 YELLOW rules — compound AND logic, ≥1 required to trigger
- [x] Implement GREEN default assignment
- [x] Handle NaN/None via `_safe()` helper (raises ValueError)
- [x] Implement Layer 2 downgrade engine — product + review rules
- [x] Implement `labeling_pipeline.py` orchestrating Layer 1 → Layer 2
- [x] Use engine output to generate `final_recommendation` labels for ML training
- [x] Write unit tests for financial engine rules and edge cases
- [x] Write unit tests for downgrade engine rules
- [x] Write unit tests for labeling pipeline orchestration
- [ ] Verify engine output cannot be overridden by ML or LLM layer

---

### Phase 4 — Model Training

**Objective:** Train baseline candidates for recommendation confidence scoring.

**Candidates:**

| Model | Type | Role |
|-------|------|------|
| XGBoost (tree booster) | Nonlinear ensemble | Primary candidate |
| LightGBM | Nonlinear ensemble | Secondary candidate |
| XGBoost (linear booster) | Linear ensemble | Fast linear baseline |

**Tasks:**
- Set fixed random seeds across NumPy, scikit-learn, and XGBoost via `Config.RANDOM_STATE`
- Create stratified 3-way split: train (60%) / validation (20%) / test (20%)
- Train all 3 candidates with default hyperparameters as baselines
- Use `eval_set` with validation data + early stopping (10 rounds) for tree-based models
- Filter invalid hyperparameters per model type automatically via `VALID_PARAMS` in `train.py`
- Log each baseline run to MLflow: model type, params, validation metrics, artifacts
- Compare baseline results in MLflow UI

**Note:** ML model output supports confidence scoring and ranking only. It does not override the deterministic financial safety logic.

**Tools:**

| Tool | Purpose |
|------|---------|
| XGBoost | Primary nonlinear baseline + linear booster variant |
| LightGBM | Secondary nonlinear baseline |
| scikit-learn | Preprocessing, pipeline |
| MLflow | Baseline run logging |

---

### Phase 5 — Hyperparameter Tuning

**Objective:** Optimize model performance while preserving fairness and robustness.

**Strategy:** Optuna with TPE sampler (Bayesian optimization) and MedianPruner for early trial termination. Each trial is logged to MLflow via Optuna's native callback.

**Approach:**
- After baseline training, identify the best-performing tunable candidate
- Run Optuna study on that candidate's search space
- Train a final model with the optimized hyperparameters
- Compare tuned vs. baseline head-to-head in MLflow

**Search Spaces:**

| Model | Parameters Tuned |
|-------|-----------------|
| XGBoost | `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_weight` |
| LightGBM | `max_depth`, `learning_rate`, `n_estimators`, `num_leaves`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| XGB-Linear | `learning_rate`, `n_estimators`, `reg_alpha`, `reg_lambda` |

**Configuration:**
- `N_TUNING_TRIALS = 50` — maximum number of trials
- `TUNING_TIMEOUT_SECONDS = 600` — safety timeout (whichever limit hits first)
- `TUNING_BACKEND = "optuna"` — set to `"none"` to skip tuning

**Tasks:**
- Define search space per model type
- Set up Optuna with MLflow callback for automatic trial logging
- Run Bayesian hyperparameter search with pruning
- Log every trial: hyperparameters, validation metrics
- Identify and tag best trial in MLflow
- Document search space and tuning strategy

**Tools:**

| Tool | Purpose |
|------|---------|
| Optuna | Bayesian optimization with TPE sampler |
| Optuna MedianPruner | Early termination of underperforming trials |
| MLflow | Trial logging via `MLflowCallback` |

---

### Phase 6 — Validation & Metrics

**Objective:** Validate model performance on unseen data using task-relevant metrics and required visualizations.

**Split Strategy:** Candidates are evaluated on the **validation set** (20%). The **test set** (20%) is used exactly once for the final selected model.

**Metrics Computed:**

| Metric | Scope | Purpose |
|--------|-------|---------|
| Accuracy | Aggregate | Overall correctness |
| F1-score (weighted) | Aggregate | Balanced performance across classes |
| ROC-AUC (weighted, OVR) | Aggregate | Discrimination ability |
| PR-AUC (weighted) | Aggregate | Performance under class imbalance |
| Precision, Recall, F1 | Per-class (GREEN, YELLOW, RED) | Identify weak classes — especially RED |

**Visualizations Generated (all logged to MLflow):**

| Visualization | Description |
|---------------|-------------|
| Confusion Matrix | 3×3 grid showing predicted vs. actual for GREEN/YELLOW/RED |
| ROC Curves | Per-class one-vs-rest curves on a single figure |
| Precision-Recall Curves | Per-class curves — exposes imbalance issues ROC hides |
| Calibration Curves | Per-class reliability diagrams for probability quality |
| Classification Report | Full precision/recall/F1 table logged as text artifact |

**Tasks:**
- Evaluate all candidates on validation set with full metric suite
- Generate all visualizations per candidate run
- Log per-class metrics individually (e.g., `RED_f1`, `GREEN_precision`)
- Apply acceptance gates — block promotion if below minimum thresholds
- Run final model on held-out test set exactly once after selection

**Tools:**

| Tool | Purpose |
|------|---------|
| sklearn.metrics | Classification metrics |
| Matplotlib | Required visualizations |
| MLflow | Metric and artifact logging |

---

### Phase 7 — Bias Detection

**Objective:** Detect performance disparities across data subgroups after model training. Bias detection is computed directly on the validation set using 17 uniquely defined slices.

**Implemented — Post-Training Bias Detection (`guards/bias_detection.py`):**
- Computes aggregate metrics (Accuracy, F1, GREEN rate).
- Evaluates Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD) via Fairlearn.
- Flags any disparities that exceed 10% for DPD/EOD or strict F1 score violations across 17 slices.
- Plugs into `run_pipeline.py` to gate any models containing bias.
- Generates a comprehensive bias report alongside the evaluation summary.

**Slice Definitions for SavVio:**

| Slice Type | Subgroups |
|------------|-----------|
| Financial | Income bands, savings band, DTI bands, savings-to-income |
| Product | Price bands, rating variance bands, review confidence bands (`rating_number`), category_name |
| Demographic | Region, employment status |

**Tools:**

| Tool | Purpose |
|------|---------|
| Fairlearn | Slice fairness analysis and Parity checks |
| Pandas | Manual slice metric computation |

---

### Phase 8 — Bias Mitigation

**Objective:** Apply robust data rebalancing and mathematical boundary shifts to correct bias.

**Implemented — Dual-Tier Mitigation System:**
1. **Pre-Training Mitigation (`data/bias_mitigation.py`)** 
   - Dynamically oversamples under-represented clusters (e.g. Unemployed/Student profiles up to 12%, Premium products up to 6%).
   - Generates synthetic cohorts (injecting "near-zero savings" vectors created from similar low-income users) to ensure model exposure.
   - Caps over-represented groups (prolific reviewers limited to 50 max predictions) preventing dataset sway.
2. **Post-Training Threshold Mitigation (`guards/bias_detection.py`)**
   - In instances where the model's post-training bias detection (Phase 7) flags a violation, Fairlearn's `ThresholdOptimizer` is automatically fired.
   - It performs mathematical boundary adjustment without recalculating XGBoost gradients, strictly forcing predictions into a passing fairness envelope.
   - Followed by a re-evaluation to confirm the newly adjusted threshold successfully corrected the disparity.

**Tools:**

| Tool | Purpose |
|------|---------|
| Fairlearn | Boundary shifts and threshold optimization (`ThresholdOptimizer`) |
| Pandas | Re-sampling, taxonomy collapsing, and synthetic cohort generation |

---

### Phase 9 — Model Selection

**Objective:** Select the final model only after both validation metrics and bias mitigation are satisfactory.

**Selection Logic:**
1. Filter out candidates that failed the bias gate
2. Rank remaining candidates by weighted F1 on the validation set
3. If no candidate passes bias, fall back to best F1 with logged warning
4. Tag selected run in MLflow as `best-model`

**Tasks:**
- Collect all candidates that passed the validation gate (Phase 6)
- Filter out any candidate that failed the bias gate (Phase 8)
- Rank remaining candidates by F1
- Select best model and document: metrics, bias results, trade-offs made
- Tag selected run in MLflow
- Log selection rationale as an MLflow artifact

**Selection rule:** Final model is never selected on aggregate accuracy alone. The bias mitigation gate must pass first.

---

### Phase 10 — Sensitivity & Explainability

**Objective:** Understand how model behavior changes with respect to hyperparameter variation and input features.

**Implemented — Optuna-Based Hyperparameter Sensitivity (`sensitivity_analysis.py`):**
- Analyze completed Optuna study trials for the tuned champion model
- Compute ranked hyperparameter importance via `optuna.importance.get_param_importances()`
- Generate importance bar plot (top-K parameters)
- Generate parameter-vs-objective scatter plots (F1 vs. each top parameter)
- Save JSON report with study name, trial count, and ranked importances
- Results saved to `reports/sensitivity/` and logged to MLflow evaluation summary
- Configurable via `Config.SENSITIVITY_ANALYSIS_ENABLED`, `Config.SENSITIVITY_MIN_COMPLETED_TRIALS`, `Config.SENSITIVITY_TOP_K_PARAMS`
- Requires minimum completed trials (default: 10) to produce meaningful analysis

**Not Yet Implemented:**
- SHAP global summary and local force plots
- LIME local explanations per class
- Feature-level instability analysis across slices

**Tools:**

| Tool | Purpose |
|------|---------|
| Optuna importance | Hyperparameter sensitivity ranking |
| Matplotlib | Importance bar plots + scatter plots |
| SHAP (planned) | Global and local feature contribution explanations |
| LIME (planned) | Local interpretable explanations |

---

### Phase 11 — Experiment Tracking

**Objective:** Track every meaningful experiment and maintain full lineage from data version to model artifact.

**Experiment Organization:**
```
Experiment: Financial_Wellbeing_Prediction
├── Run: xgboost_baseline
├── Run: lightgbm_baseline
├── Run: xgb_linear_baseline
├── Run: logistic_regression_baseline
├── Run: xgboost_tuning_trial_001 ... N  (auto-logged by Optuna)
├── Run: xgboost_tuned
└── Run: FINAL_xgboost  (held-out test evaluation)
```

**What Gets Logged Per Run:**

| Category | Items |
|----------|-------|
| Params | model_type, hyperparams, n_scenarios, random_state, label_type, num_classes |
| Aggregate Metrics | accuracy, f1_score, roc_auc, pr_auc |
| Per-Class Metrics | GREEN_f1, YELLOW_f1, RED_f1, GREEN_precision, YELLOW_recall, etc. |
| Bias Metrics | bias_gate_passed, per-slice disparity values |
| Artifacts | model binary, confusion_matrix.png, roc_curves.png, pr_curves.png, calibration_curves.png, classification_report.txt, encoder.pkl, scaler.pkl, scenarios.csv |

**Tasks:**
- Set up MLflow tracking server (local Docker with persistent volume)
- Instrument `run_pipeline.py` to auto-log: params, metrics, model artifact, data version reference
- Log bias reports and slice charts as MLflow artifacts per run
- Log all visualizations per run
- Use MLflow UI to compare all runs — capture comparison screenshot for submission
- Tag winning run as `best-model` with version label

**Tools:**

| Tool | Purpose |
|------|---------|
| MLflow Tracking | Run metadata and artifacts |
| MLflow UI | Experiment comparison and visualization |
| DVC tags / commit refs | Data lineage tie-in |

---

### Phase 12 — Model Registry Push

**Objective:** Version and store the approved model in the registry for deployment traceability and rollback capability.

**Two-stage process:**
1. **MLflow Model Registry** — Register best model with version tag for experiment tracking and comparison
2. **GCP Artifact Registry** — Push model binary for production deployment (automated via CI/CD)

**Tasks:**
- Confirm model passed all gates: validation (passed) and bias (passed)
- Register model in MLflow via `mlflow.register_model()`
- Serialize model artifact via joblib
- Push to GCP Artifact Registry
- Tag artifact with: model version, commit hash, DVC data ref, MLflow run ID
- Record rollback pointer — store previous stable model version tag

**Registry checklist:**

- [ ] Model version tag
- [ ] Commit hash
- [ ] Training data version (DVC ref)
- [ ] Validation metric report reference
- [ ] Bias analysis report reference
- [ ] Rollback pointer to previous stable model

**Tools:**

| Tool | Purpose |
|------|---------|
| MLflow Model Registry | Experiment-time model versioning |
| GCP Artifact Registry | Production deployment storage |
| Cloud IAM | Access control |

---

### Phase 13 — CI/CD Automation

**Objective:** Automate the full training → validation → bias → registry pipeline on every code change, containerized in Docker, connecting `src ↔ test ↔ DB ↔ ML`.

### Pipeline Architecture

```
GitHub Push / PR on model_pipeline/
        ↓
GitHub Actions / Cloud Build  [Dockerized]
        ├── 1. src unit tests
        ├── 2. DB connection check
        ├── 3. DVC pull (versioned data)
        ├── 4. ML training (inside Docker container)
        ├── 5. Automated validation gate
        │       └── below threshold? → BLOCK + alert
        ├── 6. Automated bias detection gate
        │       └── severe disparity? → BLOCK + alert
        ├── 7. Rollback check
        │       └── worse than previous? → BLOCK + alert
        ├── 8. Registry push (only if all gates pass)
        └── 9. Slack / email notification
```

**Gate Thresholds (configurable):**

| Gate | Metric | Threshold |
|------|--------|-----------|
| Validation | Weighted F1 | > 0.70 |
| Validation | ROC AUC | > 0.75 |
| Bias | Max F1 disparity across any slice | < 0.10 |
| Rollback | F1 vs. previous model | No decrease > 0.02 |

**Tasks:**
- Write Dockerfile to containerize full training and validation environment
- Configure GitHub Actions workflow (`.github/workflows/model_ci.yml`)
- Implement automated validation gate
- Implement automated bias gate
- Implement rollback mechanism
- Set up Slack/email notifications
- Test full end-to-end pipeline in CI environment

**Tools:**

| Tool | Purpose |
|------|---------|
| GitHub Actions | CI orchestration |
| Docker | Full pipeline containerization |
| Cloud Build | GCP-native CI/CD alternative |
| Slack / Email | Failure and completion notifications |

---

### Phase 14 — LLM Integration

**Objective:** Integrate an LLM into SavVio for two roles: (1) understanding natural language user queries and extracting product references, and (2) generating conversational purchase recommendations grounded in the deterministic engine's authoritative output.

**Architecture:**

The LLM has **two distinct roles** — it is NOT just an output wrapper:

```
User Query ("Should I buy this $1,500 laptop?")
         │
         ▼
┌─── LLM Role 1: Intent Parser ───┐
│  Parse intent → purchase_query   │  Hybrid LLM + Regex Logic
│  Extract product → "laptop"      │
└──────────────┬───────────────────┘
               │
               ▼
┌─── Product Resolver ────────────┐
│  Embed query (pgvector)         │  Cosine similarity search
│  Match → product_id: 8214        │  against products catalog
└──────────────┬──────────────────┘
               │
               ▼
    [API Layer runs Engines + ML]
    (Financial rules + XGBoost Score)
               │
               ▼
┌─── LLM Role 2: Response Gen ───┐
│  Combine Engine + ML + Context  │  Fiduciary-grounded
│  → Conversational recommend     │  conversational advice
└──────────────┬─────────────────┘
               │
               ▼
┌─── Fiduciary Guardrails ───────┐
│  G1: Color contradiction        │  Ensures LLM never
│  G2: Hallucinated figures        │  violates the authoritative
│  G3: Out-of-scope advice         │  deterministic core.
│  G4: Internal leakage            │
│  G5: Tone mismatch               │
│  G6: Length check                │
└────────────────────────────────┘
               │
               ▼
    User sees recommendation
```

**Source Files:**
- `llm/README.md` — Detailed LLM sub-package documentation
- `llm/llm_provider.py` — Strategy-pattern abstraction: `OpenRouterProvider` (Hub), `GeminiProvider`, `MockProvider`
- `llm/intent_parser.py` — Hybrid intent detection + extraction (LLM + Regex fallbacks)
- `llm/product_resolver.py` — pgvector cosine similarity search against `products` catalog
- `llm/response_generator.py` — Multi-stage conversational generation grounded in financial context
- `llm/guardrails.py` — 6 code-level safety checks ensuring fiduciary compliance

**Provider Configuration:**

| Provider | Env Var | Package | Model Default |
|----------|---------|---------|---------------|
| Mock (default) | `LLM_PROVIDER=mock` | None | Template-based |
| Google Gemini | `LLM_PROVIDER=gemini` | `google-genai` | `gemini-2.5-flash` |
| OpenAI | `LLM_PROVIDER=openai` | `openai` | `gpt-4.1` |
| Anthropic Claude | `LLM_PROVIDER=claude` | `anthropic` | `claude-4.5-sonnet` |

**Product Resolution:**
- Uses the same embedding model (`all-MiniLM-L6-v2`, 384-dim) as the data pipeline's `vector_embed.py`
- Searches `product_embeddings` table via pgvector cosine similarity
- Configurable similarity threshold (default: 0.3) and top-k results (default: 5)

**Guardrail Checks (6):**

| Check | What It Catches |
|-------|-----------------|
| **G1 — Color contradiction** | Responses that encourage a purchase when the engine labeled it RED. |
| **G2 — Hallucinated figures** | Any price or income figures mentioned that don't match the input profile. |
| **G3 — Out-of-scope advice** | Refuses to give investment, tax, or legal advice. |
| **G4 — Internal leakage** | Filters out mentions of internal rules (e.g., `RED 1`) or technical terms (`pgvector`). |
| **G5 — Tone mismatch** | Ensures empathy for RED/YELLOW and objective support for GREEN. |
| **G6 — Length check** | Strict word count limits to keep conversation concise and mobile-friendly. |

**Provider Hub:**
SavVio uses **OpenRouter** as its primary production hub, providing access to `gemini-2.0-flash` with high reliability and zero-SDK dependency. Direct SDK providers (`google-genai`, `openai`, `anthropic`) are maintained as high-performance fallbacks.

**Tasks:**
- [x] Design provider-agnostic LLM abstraction (strategy pattern)
- [x] Implement MockProvider with template-based responses
- [x] Implement GeminiProvider using `google-genai` SDK (v1.69+)
- [x] Implement OpenAI and Claude provider stubs
- [x] Build intent parser with regex (mock) and LLM-based extraction
- [x] Build product resolver using pgvector similarity search
- [x] Build response generator with template fallbacks
- [x] Implement 6 code-level guardrail checks
- [x] Design system prompt with 5 critical rules
- [x] Design intent extraction prompt template
- [x] Create GREEN/YELLOW/RED response templates with rule explanations
- [x] Version-control all prompt templates (v1.0)
- [x] Preserve backward compatibility via `prompt_engin.py` facade
- [x] Verify with Gemini API (12/12 checks passed)
- [x] Write unit tests (77 tests, all passing)
- [ ] Integrate NeMo Guardrails (deferred to deployment phase)

**Verification:**
```bash
# Run unit tests (no API calls, mock provider)
PYTHONPATH=src python -m pytest tests/llm/ -v

# Run live Gemini verification (requires GEMINI_API_KEY in .env)
python verify_llm.py
```

**Tools:**

| Tool | Purpose |
|------|---------|
| Google Gemini (`google-genai`) | Primary LLM provider for intent parsing and response generation |
| sentence-transformers | Product text embedding for pgvector search |
| pgvector | Vector similarity search for product resolution |
| Code-level guardrails | 6 safety checks (NeMo-ready interface for future integration) |

---

### Phase 15 — Monitoring & Dashboard

**Objective:** Monitor the live system for latency, drift, hallucination flags, and recommendation quality after deployment.

**Tasks:**
- Set up monitoring dashboard tracking: latency per request, refusal rate, hallucination flag rate
- Monitor ML model for data/concept drift — flag if input feature distributions shift significantly
- Monitor LLM recommendation distribution over time — detect drift in Green/Yellow/Red output ratios
- Set up alerts for:
  - Hallucination spike above threshold
  - Latency breach (response time > SLA)
  - Safety rail trigger volume increase
  - Model performance degradation vs. baseline
- Log all monitoring metrics to dashboard (Evidently / Arize)
- Trigger re-training pipeline if drift exceeds configured threshold

**Tools:**

| Tool | Purpose |
|------|---------|
| Evidently | Data and model drift detection |
| Arize | ML observability and monitoring |
| WhyLabs | Alternative monitoring platform |
| GCP Cloud Monitoring | Infrastructure and latency alerts |

---

### Phase 16 — Testing

```bash
pytest model_pipeline/tests
```

**Test coverage:**

| Test File | What It Tests |
|-----------|--------------|
| `data/test_data_loader.py` | Data loading from PostgreSQL |
| `data/test_validate_data.py` | Schema and data validation checks |
| `features/test_feature_engineering.py` | End-to-end feature engineering orchestration |
| `features/test_financial_features.py` | 6 financial feature computation |
| `features/test_product_features.py` | 7 product feature computation |
| `features/test_review_features.py` | 6 review feature computation |
| `features/test_feature_preprocessing.py` | FeaturePipeline: imputation, encoding, scaling |
| `features/test_affordability_preprocessing_consistency.py` | Cross-module consistency checks |
| `deterministic_engine/test_financial_engine.py` | Layer 1: all 4 RED rules, all 5 YELLOW rules, GREEN default, edge cases |
| `deterministic_engine/test_downgrade_engine.py` | Layer 2: product rules PR1–PR3, review rules RR1–RR3, downgrade logic |
| `deterministic_engine/test_labeling_pipeline.py` | Layer 1 + Layer 2 orchestration |
| `core_models/test_evaluate.py` | Metric computation, multi-class AUC, visualization generation |
| `core_models/test_optuna_tuner.py` | Study creation, objective functions, timeout, unsupported model error |
| `guards/test_bias_detection.py` | Fairlearn bias metrics (demographic parity, equalized odds) |

---

### Phase 17 — Operational Risks & Guardrails

### Risks

| Risk | Description |
|------|-------------|
| Financial hallucination | LLM contradicts or overrides deterministic engine output |
| Data/concept drift | Income or expense distributions shift over time |
| Sparse slice underperformance | Low-income or cold-start users receive lower quality recommendations |
| Pipeline/API failures | DVC pull errors, GCS connectivity issues, registry downtime |

### Guardrails

| Guardrail | Mechanism |
|-----------|-----------|
| Deterministic engine authority | Rule engine output is final — ML and LLM cannot override it |
| NeMo Guardrails | Safety rails enforced at LLM output boundary |
| Bias promotion gate | CI/CD blocks model push if slice disparities exceed threshold |
| Registry rollback | Previous stable model version retained; auto-revert on underperformance |
| Monitoring dashboard | Real-time alerts for latency, drift, and hallucination flags |

---

### Model Candidates — Selection Rationale

The pipeline trains and compares four model candidates: XGBoost (tree booster), LightGBM, XGBoost (linear booster), and Logistic Regression.

Two additional algorithms were evaluated during planning but excluded:

**LinearBoost** (`linearboost` package) was considered as a fast linear baseline. However, `LinearBoostClassifier` only supports binary classification. SavVio's labeling task is a 3-class problem (GREEN / YELLOW / RED), which would require wrapping LinearBoost in a `OneVsRestClassifier`. This adds complexity without a clear advantage over XGBoost's built-in linear booster (`booster='gblinear'`), which handles multi-class natively and serves the same role as a linear baseline.

**CatBoost** was considered as an alternative gradient boosting candidate alongside XGBoost and LightGBM. While CatBoost offers strong out-of-the-box performance and native categorical feature handling, its package dependency is significantly heavier (~200MB) compared to XGBoost and LightGBM. Given that our categorical features are already ordinal-encoded and our pipeline runs in Docker containers where image size affects build and deployment time, the marginal performance gain did not justify the added footprint. XGBoost and LightGBM provide sufficient coverage of the gradient boosting design space for this project's scope.

---

### Phase 18 — Dockerize Model Development

**Objective:** Containerize the model pipeline and tracking server to ensure consistent execution environments across local development and CI/CD.

**Tasks:**
- **Hardware Flexibility in Build:** Define a `Dockerfile` that supports both lightweight CPU-only execution and GPU/CUDA environments (requiring the NVIDIA Container Toolkit), managing complex base image options.
- **Multi-Service Orchestration:** Configure `docker-compose.yml` to seamlessly coordinate 5 distinct services (`postgres`, `storage` via RustFS, an ephemeral `create-bucket` container, `mlflow` server, and `ml-trainer`).
- **Startup Sequencing & Healthchecks:** Implement strict `depends_on` conditions with robust healthchecks to ensure services like RustFS and PostgreSQL are fully ready before the database logic or MLflow tracking API initiates.
- **Environment & Port Management:** Distribute dozens of environment variables across containers and carefully map ports (e.g., binding Postgres to port 5433 to prevent conflicts with the separate data pipeline's database).
- **Volume Mounting for Local Dev:** Set up pervasive volume mounts (`src`, `models`, `data`, `savviocore`, `reports`) for the `ml-trainer` container to sync local code and artifacts while maintaining connectivity to isolated tracking and storage APIs.


### Deliverable Checklist

### Professor Guidelines

- [x] Data loaded from versioned pipeline outputs (GCS via DVC)
- [x] Baseline models trained and compared
- [x] Hyperparameter tuning documented (Optuna — Bayesian optimization)
- [x] Validation metrics computed on hold-out set
- [x] Visualizations produced: confusion matrix, ROC curve, PR curve, calibration curve
- [x] Experiments tracked in MLflow with full artifact logging
- [x] Sensitivity analysis completed (Optuna-based hyperparameter importance)
- [ ] SHAP / LIME explainability analysis
- [x] Post-training slice-based bias analysis completed (Fairlearn)
- [x] Bias mitigation steps documented where disparities found
- [x] Model selection performed after bias checking
- [x] Best model pushed to MLflow Registry and GCP Artifact Registry
- [x] CI/CD pipeline: trigger → train → validate → bias → push
- [x] Automated validation gate implemented
- [x] Automated bias detection gate implemented
- [x] Notifications and alerts configured
- [x] Rollback mechanism implemented
- [x] Full pipeline containerized in Docker

### SavVio-Specific

- [x] Data source confirmed: PostgreSQL via data pipeline
- [x] Deterministic engine implemented for Green/Yellow/Red logic (compound AND rules, correlation groups)
- [x] ML model confirmed as confidence layer only — does not override engine
- [x] Optuna configured for hyperparameter search (Bayesian + pruning)
- [x] Bias detection confirmed as post-training (on validation set)
- [x] MLflow experiment tracking fully implemented
- [x] CI/CD connects src ↔ test ↔ DB ↔ ML (Dockerized)
- [x] LLM integration implemented (dual-role: intent parsing + response generation)
- [x] Prompt templates version-controlled (system_prompt v1.0, intent_prompt v1.0, response_templates v1.0)
- [x] Gemini provider verified (12/12 live API checks passed)
- [x] 6 code-level guardrails implemented and tested (77 unit tests passing)
- [x] NeMo-ready Guardrails abstraction (deferred full NeMo to production config)
- [ ] Monitoring and dashboard deployed
