# SavVio — Model Development Pipeline

**Team:** Murtaza Nipplewala, Niraj Mehta, Wen-Hsin Su, Pranathi Bombay, Rishabh Joshi, Sanjana Patnam

---

## ML Pipeline Structure

```
SavVio/
└── model_pipeline/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── model-requirements.txt
    ├── models/                         # Local dev model storage (gitignored)
    │   ├── checkpoints/
    │   ├── artifacts/
    │   └── preprocessing/
    ├── src/
    │   ├── run_pipeline.py             # End-to-end ML pipeline entrypoint
    │   ├── config.py                   # Centralized configuration
    │   ├── data/
    │   │   ├── db_loader.py            # Reads from PostgreSQL
    │   │   └── validate_data.py        # Schema validation
    │   ├── features/
    │   │   ├── feature_engineering.py  # Imputation, encoding, scaling, orchestration
    │   │   ├── affordability_features.py   # 6 computed financial features
    │   │   └── training_data_generator.py  # Scenario generation + labeling
    │   ├── deterministic_engine/
    │   │   └── decision_logic.py       # Authoritative GREEN/YELLOW/RED rules
    │   ├── core_models/
    │   │   ├── train.py                # XGBoost, LightGBM, XGB-Linear, LogReg
    │   │   └── evaluate.py             # Metrics, visualizations, MLflow logging
    │   ├── tuning/
    │   │   └── optuna_tuner.py         # Bayesian hyperparameter optimization
    │   ├── guards/
    │   │   └── bias_detection.py       # Fairlearn slice-based fairness analysis
    │   └── llm/
    │       └── prompt_engin.py         # LLM wrapping + guardrails
    └── tests/
```

---

## Quick Reference: Tools by Phase

| Phase | Primary Tools | Alternatives | CI/CD Gate |
|-------|--------------|--------------|------------|
| Data Loading | DVC, GCS, Pandas | Polars, LakeFS | Data version + schema check |
| Feature Engineering | Pandas, NumPy, scikit-learn | Polars, Feature-engine | Feature schema validation |
| Deterministic Engine | Pure Python | — | Unit tests must pass |
| Model Training | XGBoost, LightGBM, scikit-learn | — | Reproducible training run |
| Hyperparameter Tuning | Optuna | RandomizedSearchCV | Best-run tracking required |
| Validation & Metrics | sklearn.metrics, Matplotlib | Seaborn, Plotly | Minimum F1 / AUC threshold |
| Bias Detection | Fairlearn, Pandas groupby | AIF360, TFMA | Block on severe disparity |
| Bias Mitigation | Fairlearn, imbalanced-learn | scikit-learn threshold | Re-evaluate until gates pass |
| Model Selection | MLflow UI | Custom dashboards | Bias mitigation gate must pass |
| Sensitivity & Explainability | SHAP, LIME | Permutation Importance | Stability report required |
| Experiment Tracking | MLflow | Weights & Biases | Run metadata completeness |
| Model Registry Push | GCP Artifact Registry, Vertex AI | MLflow Registry | Push only on all gates pass |
| CI/CD Automation | GitHub Actions, Docker | Cloud Build, Jenkins | src ↔ test ↔ DB ↔ ML pipeline |
| LLM Wrapping | NeMo Guardrails, LangChain | Guardrails AI | Hallucination + safety gate |
| Monitoring & Dashboard | Evidently, Arize | WhyLabs, GCP Monitoring | Drift + latency alerts |

---

## Model Pipeline Execution Order

```
1.  Load Data (PostgreSQL via data pipeline)
        ↓
2.  Feature Engineering + Scenario Generation
        ↓
3.  Deterministic Engine → GREEN / YELLOW / RED labels
        ↓
4.  3-Way Stratified Split (train 60% / val 20% / test 20%)
        ↓
5.  Baseline Training (XGBoost, LightGBM, XGB-Linear, Logistic Regression)
        ↓
6.  Hyperparameter Tuning (Optuna on best baseline)
        ↓
7.  Validation on Validation Set (per-candidate metrics + visualizations)
        ↓
8.  Bias Detection — Post-Training (slice analysis on validation set)
        ↓
9.  Model Selection (F1 ranking + bias gate)
        ↓
10. Final Evaluation on Held-Out Test Set
        ↓
11. Sensitivity & Explainability (SHAP / LIME)
        ↓
12. Experiment Tracking (MLflow — all runs, artifacts, comparisons)
        ↓
13. Model Registry Push (MLflow registry + GCP Artifact Registry)
        ↓
14. CI/CD Automation (Dockerized)
        ↓
15. LLM Wrapping + NeMo Guardrails
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
14. [Phase 14 — LLM Wrapping & Guardrails](#phase-14--llm-wrapping--guardrails)
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

**Objective:** Transform raw DB tables into a model-ready feature matrix with deterministic GREEN/YELLOW/RED labels.

**Pipeline:**
1. `generate_training_data()` — Load data, create scenarios via stratified sampling, label with deterministic engine
2. `transform_features()` — Impute missing values, encode categoricals, scale numerics, drop non-feature columns
3. `build_feature_matrix()` — Orchestrator that calls 1 then 2 and returns `(X, y, scenarios_raw)`

**Feature Groups:**

| Group | Features | Source |
|-------|----------|--------|
| Financial (DB) | `discretionary_income`, `debt_to_income_ratio`, `saving_to_income_ratio`, `monthly_expense_burden_ratio`, `emergency_fund_months` | financial_profiles table |
| Product (DB) | `price`, `average_rating`, `rating_number`, `rating_variance` | products table |
| Computed (6) | `affordability_score`, `price_to_income_ratio`, `residual_utility_score`, `savings_to_price_ratio`, `net_worth_indicator`, `credit_risk_indicator` | affordability_features.py |
| Categorical | `employment_status`, `has_loan`, `region` | financial_profiles table |

**Scenario Generation:**
- Stratified sampling across 9 (income × price) bracket cells for balanced representation
- 50,000 scenarios by default (configurable via `Config.N_SCENARIOS`)
- Each scenario = one (user, product) pair with computed features + deterministic label

**Tasks:**
- Handle missing values — median imputation for financial numerics, 0 for rating_variance, 'Unknown' for categoricals
- Encode categorical features via OrdinalEncoder (saved as artifact for inference)
- Scale numeric features via StandardScaler (saved as artifact for inference)
- Save raw scenarios as versioned CSV artifact

**Tools:**

| Tool | Purpose |
|------|---------|
| Pandas / NumPy | Feature construction |
| scikit-learn | OrdinalEncoder, StandardScaler |

---

### Phase 3 — Deterministic Decision Engine

**Location:** `deterministic_engine/decision_logic.py`

The deterministic engine is a pure-financial multi-condition labeling system. It is built **before** model training because it generates the labels (GREEN/YELLOW/RED) that the ML model trains on. Its output is authoritative — neither the ML model nor the LLM layer can override it.

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

Each rule crosses at least 2 correlation groups and includes a `price_to_income_ratio` escape hatch so trivial purchases never trigger RED. RED returns immediately — no further evaluation.

| Rule | Groups Crossed | Condition |
|------|----------------|-----------|
| RED 1 — Can't afford from any angle | 1 + 2 | `affordability_score < 0` AND `savings_to_price_ratio < 1.5` AND `residual_utility_score < 1.0` AND `price_to_income_ratio > 0.10` |
| RED 2 — Maxed budget, significant purchase | 3 + 1 + 2 | `monthly_expense_burden_ratio > 0.80` AND `price_to_income_ratio > 0.20` AND `emergency_fund_months < 3.0` AND `savings_to_price_ratio < 3.0` |
| RED 3 — Underwater, no surplus | 4 + 1 + 2 | `net_worth_indicator < -2.0` AND `affordability_score < 0` AND `price_to_income_ratio > 0.15` AND `emergency_fund_months < 3.0` |
| RED 4 — Paycheck-to-paycheck | 2 + 3 | `emergency_fund_months < 1.0` AND `residual_utility_score < 0.5` AND `debt_to_income_ratio > 0.30` AND `price_to_income_ratio > 0.10` |

### YELLOW Rules (5 compound AND rules, require ≥2 to trigger)

Each rule crosses at least 2 correlation groups. YELLOW triggers when 2 or more rules fire.

| Rule | Groups Crossed | Condition |
|------|----------------|-----------|
| YELLOW 1 — Income pressure | 1 + 2 | `affordability_score < 0` AND `price_to_income_ratio > 0.25` AND `savings_to_price_ratio < 10.0` |
| YELLOW 2 — Savings strain | 2 + 1 | `savings_to_price_ratio < 5.0` AND `residual_utility_score < 3.0` AND `price_to_income_ratio > 0.10` |
| YELLOW 3 — Debt stress | 3 + 2 | `debt_to_income_ratio > 0.30` AND `emergency_fund_months < 4.0` AND `price_to_income_ratio > 0.10` |
| YELLOW 4 — Low resilience | 2 + 1 | `emergency_fund_months < 3.0` AND `saving_to_income_ratio < 0.25` AND `affordability_score < 0` |
| YELLOW 5 — Weak profile | 4 + 1 + 2 | `credit_risk_indicator < 0.35` AND `net_worth_indicator < 1.0` AND `price_to_income_ratio > 0.15` AND `savings_to_price_ratio < 10.0` |

### GREEN — Default

No RED rules fired and fewer than 2 YELLOW rules accumulated.

### Edge Cases

- Missing financial fields → default YELLOW
- NaN or None values → safe defaults (0.0 for most, 0.5 for credit_risk_indicator)
- Conflicting rules → more conservative class wins

### Tasks

- [x] Implement RED rules — compound AND logic with PIR escape hatch
- [x] Implement YELLOW rules — compound AND logic, ≥2 required to trigger
- [x] Implement GREEN default assignment
- [x] Handle NaN/None via `_safe()` helper
- [x] Use engine output to generate labels for ML model training
- [ ] Write unit tests for all rule conditions and edge cases
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
| Logistic Regression | Linear model | Sanity-check baseline |

**Tasks:**
- Set fixed random seeds across NumPy, scikit-learn, and XGBoost via `Config.RANDOM_STATE`
- Create stratified 3-way split: train (60%) / validation (20%) / test (20%)
- Train all 4 candidates with default hyperparameters as baselines
- Use `eval_set` with validation data + early stopping (10 rounds) for tree-based models
- Filter invalid hyperparameters per model type automatically
- Log each baseline run to MLflow: model type, params, validation metrics, artifacts
- Compare baseline results in MLflow UI

**Note:** ML model output supports confidence scoring and ranking only. It does not override the deterministic financial safety logic.

**Tools:**

| Tool | Purpose |
|------|---------|
| XGBoost | Primary nonlinear baseline |
| LightGBM | Secondary nonlinear baseline |
| scikit-learn | Logistic Regression, preprocessing, pipeline |
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

**Objective:** Detect performance disparities across meaningful data subgroups after model training. Bias detection is performed **post-training** — run on validation set predictions after model fitting is complete.

**Tasks:**
- Define all slices in `Config.SENSITIVE_FEATURES`
- Collect model predictions and ground truth per slice on validation set
- Compute per-slice metrics: Accuracy, F1, AUC for each subgroup
- Compare per-slice vs. aggregate metrics — flag disparities above configured threshold
- Generate bias report: F1 bar chart per slice, disparity summary table
- Log bias report and visualizations to MLflow
- Document all detected disparities before moving to mitigation

**Slice Definitions for SavVio:**

| Slice Type | Subgroups |
|------------|-----------|
| Financial | Income bands, DTI bands, savings-to-income, emergency fund runway |
| Product | Price bands, rating variance bands, review confidence bands (`rating_number`) |
| Demographic | Region, employment status |

**Tools:**

| Tool | Purpose |
|------|---------|
| Fairlearn | Slice fairness analysis via `MetricFrame` |
| AIF360 | Alternative fairness toolkit |
| Pandas groupby | Manual slice metric computation |

---

### Phase 8 — Bias Mitigation

**Objective:** Apply mitigation strategies to address detected bias and re-evaluate until disparities fall within acceptable thresholds.

**Tasks:**
- Review bias report from Phase 7 — identify which slices exceed disparity threshold
- Apply one or more mitigation strategies:
  - **Re-weighting** — assign higher loss weights to underrepresented groups
  - **Controlled re-sampling** — oversample sparse slices in training data
  - **Decision threshold adjustment** — set different classification thresholds per slice
  - **Stratified re-training** — retrain with stratified splits enforcing slice balance
- Re-run bias detection (Phase 7) after mitigation
- Compare pre- and post-mitigation disparity metrics
- Document trade-offs made (e.g., slight drop in aggregate accuracy for fairness gain)
- If disparity persists beyond threshold → block model promotion via CI/CD gate

**Tools:**

| Tool | Purpose |
|------|---------|
| Fairlearn | Fairness constraints and threshold optimization |
| imbalanced-learn | Re-sampling strategies |
| scikit-learn | Threshold adjustment per class |

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

**Objective:** Understand how model behavior changes with respect to input features and hyperparameter variation.

**Tasks:**
- Compute global feature importance from XGBoost built-in scores
- Run SHAP on selected model — generate global summary plot and local force plots
- Run LIME on 3–5 representative predictions per class (Green / Yellow / Red)
- Generate hyperparameter sensitivity curves (F1 vs. key hyperparameters) from Optuna study
- Identify features that cause instability across financial and product slices
- Document top 5–10 driving features for Green/Yellow/Red classification
- Log all SHAP plots, LIME explanations, and sensitivity charts to MLflow
- Write stability notes flagging inconsistent features across slices

**Tools:**

| Tool | Purpose |
|------|---------|
| SHAP | Global and local feature contribution explanations |
| LIME | Local interpretable explanations |
| Permutation Importance | Fast sensitivity baseline |

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
- Confirm model passed all gates: validation ✅ and bias ✅
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

### Phase 14 — LLM Wrapping & Guardrails

**Objective:** Wrap ML and deterministic outputs with an LLM for natural language delivery, enforced by safety guardrails and prompt engineering.

**Tasks:**
- Design system prompt template including: financial signals, decision class (Green/Yellow/Red), product context
- Implement `llm_wrapper.py` — takes deterministic engine output + ML confidence → generates natural language recommendation
- Enforce that LLM output never contradicts the deterministic engine color class
- Integrate NeMo Guardrails — define rails for: hallucinated financial figures, out-of-scope advice, unsafe outputs
- Test guardrails with adversarial prompts — confirm rails block unsafe completions
- Version-control all prompt templates alongside model artifacts
- Document all prompt engineering decisions and guardrail configuration

**Tools:**

| Tool | Purpose |
|------|---------|
| NeMo Guardrails | LLM safety and hallucination prevention |
| LangChain / LlamaIndex | Prompt orchestration |
| Evidently / Arize | Output monitoring and drift detection |

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
| `test_load_data.py` | Schema checks, data loading, join logic |
| `test_features.py` | Feature construction, encoding, scaling, scenario generation |
| `test_train.py` | All 4 model types, param filtering, early stopping, seed reproducibility |
| `test_evaluate.py` | Metric computation, multi-class AUC, visualization generation |
| `test_bias_slicing.py` | Slice metric computation, disparity detection, pass/fail logic |
| `test_optuna_tuner.py` | Study creation, objective functions, timeout, unsupported model error |
| `test_sensitivity.py` | SHAP output shape, LIME stability |
| `test_registry.py` | Registry push, rollback pointer |
| `test_decision_logic.py` | All 4 RED rules, all 5 YELLOW rules, GREEN default, edge cases |
| `test_llm_wrapper.py` | Guardrail enforcement, prompt output validation |

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
- [ ] Sensitivity and explainability analysis completed (SHAP / LIME)
- [ ] Post-training slice-based bias analysis completed
- [ ] Bias mitigation steps documented where disparities found
- [x] Model selection performed after bias checking
- [ ] Best model pushed to GCP Artifact Registry or Vertex AI
- [ ] CI/CD pipeline: trigger → train → validate → bias → push
- [ ] Automated validation gate implemented
- [ ] Automated bias detection gate implemented
- [ ] Notifications and alerts configured
- [ ] Rollback mechanism implemented
- [x] Full pipeline containerized in Docker

### SavVio-Specific

- [x] Data source confirmed: PostgreSQL via data pipeline
- [x] Deterministic engine implemented for Green/Yellow/Red logic (compound AND rules, correlation groups)
- [x] ML model confirmed as confidence layer only — does not override engine
- [x] Optuna configured for hyperparameter search (Bayesian + pruning)
- [x] Bias detection confirmed as post-training (on validation set)
- [x] MLflow experiment tracking fully implemented
- [ ] CI/CD connects src ↔ test ↔ DB ↔ ML (Dockerized)
- [ ] LLM wrapping implemented with NeMo Guardrails
- [ ] Prompt templates version-controlled
- [ ] Monitoring and dashboard deployed
