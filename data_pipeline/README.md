# SavVio - Data Pipeline Phase
**Team Members:** Murtaza Nipplewala, Niraj Mehta, Wen-Hsin Su, Pranathi Bombay, Rishabh Joshi, Sanjana Patnam

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     SavVio Data Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. SETUP                                                       │
│     └── Environment, Docker, Git, DVC init                      │
│                        ↓                                        │
│  2. DATA COLLECTION (Planning)                                  │
│     └── Identify sources, document requirements, privacy        │
│                        ↓                                        │
│  3. DATA INGESTION                                              │
│     ├── Load Financial data                                     │
│     ├── Load Product data                    [parallel]         │
│     └── Load Review data                     [parallel]         │
│                        ↓                                        │
│  4. VERSION RAW DATA (DVC Checkpoint #1)                        │
│     └── dvc add data/raw/                                       │
│                        ↓                                        │
│  5. SCHEMA & STATISTICS GENERATION                              │
│     └── Define expected structure, compute baseline stats       │
│                        ↓                                        │
│  6. RAW DATA VALIDATION                                         │
│     └── Validate raw data against schema                        │
│                        ↓                                        │
│  7. ANOMALY DETECTION & ALERTS                                  │
│     └── Detect outliers, missing values, trigger alerts         │
│                        ↓                                        │
│  8. DATA PREPROCESSING                                          │
│     └── Clean, transform, standardize                           │
│                        ↓                                        │
│  9. PROCESSED DATA VALIDATION                                   │
│     └── Validate preprocessing didn't break data                │
│                        ↓                                        │
│  10. VERSION PROCESSED DATA (DVC Checkpoint #2)                 │
│      └── dvc add data/processed/                                │
│                        ↓                                        │
│  11. FEATURE ENGINEERING                                        │
│      └── Create derived features (RUS, affordability, sentiment)│
│                        ↓                                        │
│  12. FEATURE VALIDATION                                         │
│      └── Validate feature calculations and ranges               │
│                        ↓                                        │
│  13. VERSION FEATURES (DVC Checkpoint #3)                       │
│      └── dvc add data/features/ (features.dvc)                 │
│                        ↓                                        │
│  14. LOAD TO DATABASE                                           │
│      ├── PostgreSQL (financial, product, review data)           │
│      └── pgvector (product embeddings for RAG)                  │
│                        ↓                                        │
│  15. BIAS DETECTION & MITIGATION                                │
│      └── Slice analysis on features, fairness checks            │
│                        ↓                                        │
│  16. PIPELINE ORCHESTRATION (Airflow DAG)                       │
│      └── Connect all tasks, set dependencies                    │
│                        ↓                                        │
│  17. TESTING                                                    │
│      └── Unit tests for each component                          │
│                        ↓                                        │
│  18. TRACKING, LOGGING & MONITORING                             │
│      └── Logging, metrics, dashboards                           │
│                        ↓                                        │
│  19. PIPELINE OPTIMIZATION                                      │
│      └── Gantt analysis, parallelization                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Tools by Phase

| Phase                | Primary Tools                 | Alternatives                            | Airflow Integration            |
| -------------------- | ----------------------------- | --------------------------------------- | ------------------------------ |
| Setup                | Docker, Git, DVC              | —                                       | —                              |
| Data Collection      | Documentation                 | —                                       | —                              |
| Ingestion            | Pandas, API clients           | Polars                                  | PythonOperator                 |
| Schema/Stats         | Great Expectations            | Pandera, ydata-profiling, custom Python | PythonOperator                 |
| Raw Validation       | Great Expectations            | Pandera, Pydantic, custom validators    | PythonOperator                 |
| Anomaly Detection    | Great Expectations, Evidently | Custom Python (IQR/z-score)             | PythonOperator + EmailOperator |
| Preprocessing        | Pandas, DuckDB                | Polars                                  | PythonOperator                 |
| Processed Validation | Great Expectations            | Pandera, custom validators              | PythonOperator                 |
| Features             | Pandas, NumPy                 | Polars                                  | PythonOperator                 |
| Feature Validation   | Great Expectations            | Pandera, custom validators              | PythonOperator                 |
| Versioning           | DVC, GCP Cloud Storage        | —                                       | BashOperator                   |
| Load to Database     | SQLAlchemy, psycopg2          | pandas.to_sql                           | PythonOperator                 |
| Embeddings           | Sentence-Transformers, OpenAI | LangChain embeddings                    | PythonOperator                 |
| Bias Detection       | Fairlearn                     | Custom Pandas slicing, AIF360           | PythonOperator                 |
| Orchestration        | Apache Airflow                | —                                       | Native                         |
| Testing              | pytest                        | unittest                                | —                              |
| Monitoring           | Python logging, Airflow UI    | GCP Cloud Logging, Grafana              | Native                         |
| Optimization         | Airflow Gantt                 | cProfile                                | Native                         |

---

## Table of Contents

1. [Phase 1: Project Setup & Environment Configuration](#phase-1-project-setup--environment-configuration)
2. [Phase 2: Data Collection & Planning](#phase-2-data-collection--planning)
3. [Phase 3: Data Ingestion](#phase-3-data-ingestion)
4. [Phase 4: Version Raw Data (DVC Checkpoint #1)](#phase-4-version-raw-data-dvc-checkpoint-1)
5. [Phase 5: Schema & Statistics Generation](#phase-5-schema--statistics-generation)
6. [Phase 6: Raw Data Validation](#phase-6-raw-data-validation)
7. [Phase 7: Anomaly Detection & Alerts](#phase-7-anomaly-detection--alerts)
8. [Phase 8: Data Preprocessing & Transformation](#phase-8-data-preprocessing--transformation)
9. [Phase 9: Processed Data Validation](#phase-9-processed-data-validation)
10. [Phase 10: Version Processed Data (DVC Checkpoint #2)](#phase-10-version-processed-data-dvc-checkpoint-2)
11. [Phase 11: Feature Engineering](#phase-11-feature-engineering)
12. [Phase 12: Feature Validation](#phase-12-feature-validation)
13. [Phase 13: Version Features (DVC Checkpoint #3)](#phase-13-version-features-dvc-checkpoint-3)
14. [Phase 14: Load to Database](#phase-14-load-to-database)
15. [Phase 15: Bias Detection & Mitigation](#phase-15-bias-detection--mitigation)
16. [Phase 16: Pipeline Orchestration (Airflow DAGs)](#phase-16-pipeline-orchestration-airflow-dags)
17. [Phase 17: Testing](#phase-17-testing)
18. [Phase 18: Tracking, Logging & Monitoring](#phase-18-tracking-logging--monitoring)
19. [Phase 19: Pipeline Optimization](#phase-19-pipeline-optimization)

---

## Phase 1: Project Setup & Environment Configuration

### Objective

Establish the foundational project structure, dependencies, and development environment before any data work begins.

### Steps

1. **Create folder structure** following the required format (actual implementation layout):

   ```
   SavVio/
   ├── data_pipeline/
   │   ├── README.md              # This file
   │   ├── SETUP_AND_RUN.md       # Setup and run instructions (reproducibility)
   │   ├── data-requirements.txt  # Python dependencies (or use repo root requirements.txt)
   │   ├── config/                # Configuration (Airflow, Token, GCP)
   │   ├── logs/                  # Pipeline execution logs
   │   ├── tests/                 # Unit tests (pytest)
   │   │   ├── ingestion/
   │   │   ├── preprocess/
   │   │   ├── features/
   │   │   ├── validation/
   │   │   ├── database/
   │   │   ├── bias/
   │   │   └── test_data_pipeline_airflow.py
   │   └── dags/                  # Airflow DAG and pipeline code
   │       ├── data_pipeline_airflow.py   # Main DAG definition
   │       ├── data/
   │       │   ├── raw/           # Raw data (financial(csv), product(jsonl), review(jsonl))
   │       │   ├── processed/     # Preprocessed outputs
   │       │   ├── features/      # Feature-engineered outputs
   │       │   ├── raw.dvc        # DVC pointer for raw (versioned in Git)
   │       │   ├── processed.dvc  # DVC pointer for processed
   │       │   └── features.dvc   # DVC pointer for features
   │       └── src/
   │           ├── ingestion/     # Data acquisition (GCS, APIs)
   │           │   ├── __init__.py
   │           │   ├── api_loader.py
   │           │   ├── config.py
   │           │   ├── gcs_loader.py
   │           │   └── run_ingestion.py
   │           ├── preprocess/    # Cleaning and transformation
   │           │   ├── __init__.py
   │           │   ├── financial.py
   │           │   ├── product.py
   │           │   ├── review.py
   │           │   ├── run_preprocessing.py
   │           │   └── utils.py
   │           ├── features/      # Feature engineering
   │           │   ├── __init__.py
   │           │   ├── financial_features.py
   │           │   ├── product_review_features.py
   │           │   ├── run_features.py
   │           │   └── utils.py
   │           ├── validation/   # Schema, stats, anomaly checks (Great Expectations)
   │           │   ├── __init__.py
   │           │   ├── anomaly/
   │           │   │   ├── __init__.py
   │           │   │   ├── anomaly_validator.py
   │           │   │   └── detectors.py
   │           │   ├── run_validation.py
   │           │   ├── validate/
   │           │   │   ├── __init__.py
   │           │   │   ├── feature_validator.py
   │           │   │   ├── processed_validator.py
   │           │   │   └── raw_validator.py
   │           │   └── validation_config.py
   │           ├── database/      # PostgreSQL and pgvector load
   │           │   ├── __init__.py
   │           │   ├── db_connection.py
   │           │   ├── db_schema.py
   │           │   ├── run_database.py
   │           │   ├── upload_to_db.py
   │           │   └── vector_embed.py
   │           └── bias/          # Bias detection (data slicing)
   │               ├── __init__.py
   │               ├── financial_bias.py
   │               ├── product_bias.py
   │               ├── review_bias.py
   │               ├── run_bias.py
   │               └── utils.py
   ```

2. **Configure Docker environment for Airflow**
   - Use official Apache Airflow Docker Compose setup 
   - Follow SETUP_AND_RUN

3. **Initialize version control**
   - Git repository setup
   - Create `.gitignore` (exclude data files, logs, credentials, `__pycache__`)
   - Initialize DVC: `dvc init`
   - Configure DVC remote: `dvc remote add -d gcs gs://savvio-data-bucket`

4. **Configure ENV and database connections**
   - Copy `.env.example` to `.env`
   - Follow SETUP_AND_RUN to configure ENV and database connections
   - Local (Development): Local PostgreSQL instance
   - Cloud (Production): GCP Cloud SQL for PostgreSQL


---

## Phase 2: Data Collection & Planning

### Objective

Identify, understand, and document the data sources needed for SavVio's purchase guardrail functionality. This phase is about planning and documentation before actual data ingestion.

### Steps

1. **Translate user needs into data needs**
   - Users: Consumers making purchase decisions
   - User Need: Make informed, responsible purchase decisions
   - System Need: Financial health data + Product information + Product reviews

2. **Document data privacy measures**
   - Masking/hashing user identifiers
   - Encryption for financial snapshots
   - Read-only access (no transactional capabilities)
   - Compliance with data privacy principles
   - Anonymized review data (no PII)

3. **Create Data Card documentation**

   **Financial Data (from Kaggle):**
   | Field | Description |
   |-------|-------------|
   | user_id | Unique user identifier |
   | age | Age of individual |
   | gender | Gender |
   | education_level | Highest education level |
   | employment_status | Employment type |
   | job_title | Job title or role |
   | monthly_income_usd | Approx. monthly income in USD |
   | monthly_expenses_usd | Approx. monthly expenses in USD |
   | savings_usd | Total savings |
   | has_loan | Whether individual has a loan |
   | loan_type | Type of loan |
   | loan_amount_usd | Loan principal amount |
   | loan_term_months | Duration of loan |
   | monthly_emi_usd | Monthly installment (EMI) |
   | loan_interest_rate_pct | Interest rate on loan (%) |
   | debt_to_income_ratio | Ratio of debt payments to income |
   | credit_score | Synthetic credit score |
   | savings_to_income_ratio | Ratio of savings to annual income |
   | region | Geographic region |
   | record_date | Record creation date |

   **Product Data (from Amazon Reviews'23):**
   | Field | Description |
   |-------|-------------|
   | main_category | Main category of the product |
   | title | Name of the product |
   | average_rating | Rating of the product |
   | rating_number | Number of ratings |
   | features | Bullet-point features |
   | description | Description of product |
   | price | Price in USD |
   | images | Product images |
   | videos | Product videos |
   | store | Store name |
   | categories | Hierarchical categories |
   | details | Product details |
   | parent_asin | Parent ID of product |
   | bought_together | Recommended bundles |

   **Review Data (from Amazon Reviews'23):**
   | Field | Description |
   |-------|-------------|
   | rating | Rating of the product (1-5) |
   | title | Title of the review |
   | text | Text body of review |
   | images | Images posted by user |
   | asin | ID of the product |
   | parent_asin | Parent ID of the product |
   | user_id | ID of reviewer |
   | timestamp | Time of review |
   | verified_purchase | User purchase verification |
   | helpful_vote | Helpful votes |




---

## Phase 3: Data Ingestion

### Objective

Download data from external sources and load it into the pipeline system in a consistent, reproducible manner.

### Steps

1. **Implement shared utilities** (`src/ingestion/config.py`, `src/ingestion/api_loader.py`)
   - Logging, config variables, API wrapper functions.

2. **Implement generic loaders** (`src/ingestion/gcs_loader.py`)
   - `download_from_gcs()` — Fetches files from cloud storage.
   - Saves to `data/raw/` path based on provided keys.

3. **Run ingestion pipeline** (`src/ingestion/run_ingestion.py`)
   - Connects to sources (GCS, APIs) sequentially or in parallel.
   - Fetches and stores raw data.

4. **Store raw data in original format**
   - Save to `data/raw/financial.csv`
   - Save to `data/raw/products.jsonl`
   - Save to `data/raw/reviews.jsonl`

5. **Log ingestion metadata**
   - Timestamp, record counts, checksums, errors.

---

## Phase 4: Version Raw Data (DVC Checkpoint #1)

### Objective

Version control the raw ingested data to ensure reproducibility and enable rollback if needed. Data is versioned using DVC with separate `.dvc` pointer files for raw, processed, and features (history maintained in Git).

### Steps

1. **Add raw data to DVC tracking** (run from `data_pipeline/dags/data`)

   ```bash
   cd data_pipeline/dags/data
   dvc add raw
   ```

   This creates `raw.dvc` (and updates `.gitignore` for the raw directory).

2. **Commit .dvc files to Git**

   ```bash
   git add raw.dvc
   git commit -m "Add raw data v1.0"
   ```

3. **Push data to remote storage**

   ```bash
   dvc push
   ```

4. **Tag the version**
   ```bash
   git tag -a "data-raw-v1.0" -m "Initial raw data ingestion"
   ```


### Why Version Here?

- Captures original data before any modifications
- Enables rollback if preprocessing introduces errors
- Provides "source of truth" for debugging

---

## Phase 5: Data Schema & Statistics Generation

### Objective

Define the expected structure (data schema) of your datasets and compute baseline statistics.

> **Note:** "Data Schema" = expected structure/rules for a dataset (columns, types, constraints).  
> "Database Schema" = tables, columns, relationships in a database (covered in Phase 14).

### Steps

1. **Define data schema (expected structure)**

   **Financial Data Schema:**
   | Column | Type | Nullable | Constraints |
   |--------|------|----------|-------------|
   | monthly_income | float | No | >= 0 |
   | rent | float | Yes | >= 0 |
   | recurring_bills | float | Yes | >= 0 |
   | savings_balance | float | Yes | >= 0 |
   | debt_obligations | float | Yes | >= 0 |

   **Product Data Schema:**
   | Column | Type | Nullable | Constraints |
   |--------|------|----------|-------------|
   | product_name | string | No | non-empty |
   | category | string | No | valid category list |
   | price | float | No | > 0 |
   | specifications | string | Yes | — |
   | description | string | Yes | — |

   **Review Data Schema:**
   | Column | Type | Nullable | Constraints |
   |--------|------|----------|-------------|
   | product_id | integer/string | No | foreign key valid |
   | reviewer_id | string | No | non-empty |
   | rating | float | No | 1-5 range |
   | text | string | Yes | — |
   | helpful_count | integer | Yes | >= 0 |
   | date | timestamp | Yes | valid date |

2. **Compute baseline statistics**
   - Numeric: min, max, mean, median, std, percentiles
   - Categorical: unique values, frequency distribution
   - All columns: null percentage, data type distribution

3. **Create expectation suites (Great Expectations)**
   - `expect_column_to_exist`
   - `expect_column_values_to_be_of_type`
   - `expect_column_values_to_not_be_null`
   - `expect_column_values_to_be_between`
   - `expect_column_values_to_be_in_set`


---

## Phase 6: Raw Data Validation

### Objective

Validate raw ingested data against the schema defined in Phase 5 to catch ingestion errors early.

### Steps

1. **Run validation checkpoint on raw data**
   - Load expectation suite
   - Execute against `data/raw/` files
   - Generate validation results

2. **Validation rules**

   **Financial Data:**
   - All expected columns exist
   - Data types are correct
   - Required fields not null
   - Values within expected ranges

   **Product Data:**
   - All expected columns exist
   - Price is positive
   - Product name not empty

   **Review Data:**
   - All expected columns exist
   - Rating between 1-5
   - Product ID references valid product
   - Reviewer ID not empty

3. **Handle validation failures**
   - Log errors with details
   - Quarantine invalid records to `data/quarantine/`
   - Generate HTML report (Great Expectations Data Docs)
   - Critical failures halt pipeline

4. **Store results**
   - `logs/validation/raw_validation_YYYYMMDD.json`


---

## Phase 7: Anomaly Detection & Alerts

### Objective

Detect data anomalies (outliers, suspicious patterns) and trigger alerts when issues are found.

### Steps

1. **Define anomaly rules**

   **Financial Data:**
   | Anomaly | Detection Method |
   |---------|------------------|
   | Income = 0 | Rule-based |
   | Expenses > 2x income | Rule-based |
   | Negative monetary values | Rule-based |
   | Extreme outliers | IQR or z-score |

   **Product Data:**
   | Anomaly | Detection Method |
   |---------|------------------|
   | Price <= 0 | Rule-based |
   | Price > $100,000 | Threshold |
   | Missing name with valid price | Rule-based |

   **Review Data:**
   | Anomaly | Detection Method |
   |---------|------------------|
   | Rating outside 1-5 | Rule-based |
   | Product ID doesn't exist | Referential check |
   | Duplicate reviews | Deduplication |
   | Extreme helpful counts | Outlier detection |

2. **Implement detection**
   - Statistical: IQR method, z-score
   - Rule-based: business logic checks

3. **Configure alerts**

   | Severity | Condition      | Action       |
   | -------- | -------------- | ------------ |
   | INFO     | Minor outliers | Log only     |
   | WARNING  | >5% issues     | Email alert  |
   | CRITICAL | Data corrupted | Email + halt |

4. **Quarantine suspicious records**


---

## Phase 8: Data Preprocessing & Transformation

### Objective

Clean, transform, and standardize validated data into a consistent format ready for feature engineering.

### Steps

1. **Data cleaning**

   **Financial Data:**
   - Handle missing values (impute median or flag)
   - Standardize transaction categories
   - Remove duplicates

   **Product Data:**
   - Flatten nested JSONL objects if present
   - Remove duplicates
   - Standardize price format
   - Clean product names

   **Review Data:**
   - Flatten nested JSONL objects if present
   - Remove duplicate reviews
   - Clean text (trim whitespace, handle special chars)
   - Validate ratings are in 1-5 range
   - Handle missing helpful_count

2. **Data transformation**

   **Financial Data:**
   - Convert to monthly format (annual ÷ 12, weekly × 4.33)
   - Categorize expenses
   - Calculate total fixed expenses

   **Product Data:**
   - Standardize category names
   - Clean descriptions for embedding

   **Review Data:**
   - Normalize text for processing
   - Aggregate ratings by product

3. **Incremental Merging (Out-of-Core)**
   - To support continuous daily ingestion without blowing up Memory, new batched records are merged with the existing historical dataset using **DuckDB**.
   - Records with matching keys (`asin`, `user_id`) in both files are updated/replaced.
   - New records are appended. 
   - Operations performed entirely out-of-core utilizing `/tmp` disk spilling to stay within strict Docker RAM limits.

4. **Save processed data**
   - `data/processed/financial_processed.csv`
   - `data/processed/products_processed.csv`
   - `data/processed/reviews_processed.csv`


---

## Phase 9: Processed Data Validation

### Objective

Validate that preprocessing transformations didn't break the data or introduce errors.

### Steps

1. **Define processed data expectations**
   - All original records accounted for (minus quarantined)
   - No new null values introduced
   - Transformations applied correctly (e.g., monthly format)
   - No duplicate records

2. **Run validation checkpoint**
   - Validate `data/processed/` files
   - Compare record counts: raw vs processed
   - Verify transformation logic

3. **Specific checks for SavVio**
   - Financial values in monthly format
   - Expense categories standardized
   - Product prices positive and reasonable
   - Review ratings normalized and valid

4. **Handle failures**
   - Log transformation errors
   - Alert if significant data loss
   - Option to rollback to raw data


---

## Phase 10: Version Processed Data (DVC Checkpoint #2)

### Objective

Version control the processed data to track transformations.

### Steps

1. **Add processed data to DVC** (run from `data_pipeline/dags/data`)

   ```bash
   cd data_pipeline/dags/data
   dvc add processed
   ```

   This creates/updates `processed.dvc`.

2. **Commit and push**

   ```bash
   git add processed.dvc
   git commit -m "Add processed data v1.0"
   dvc push
   ```

3. **Tag the version**
   ```bash
   git tag -a "data-processed-v1.0" -m "Processed data"
   ```

### Why Version Here?

- Captures cleaned data before feature engineering
- If features have bugs, no need to re-preprocess
- Enables raw vs processed comparison

---

## Phase 11: Feature Engineering

### Objective

Create meaningful features within each data track — financial health features per user and quality features per product. Affordability metrics (which require both user and product data) are computed at inference time by the Deterministic Financial Logic Engine, not pre-computed in the pipeline.

### Steps

1. **Financial health features** (`financial_features.py`)

   Input: `data/processed/financial_preprocessed.csv`
   Output: `data/features/financial_featured.csv`

   | Feature                        | Formula                    | Purpose                           |
   | ------------------------------ | -------------------------- | --------------------------------- |
   | `discretionary_income`         | income - (expenses + emi)  | Available money after obligations |
   | `debt_to_income_ratio`         | emi / income               | Debt burden indicator             |
   | `savings_to_income_ratio`      | savings / income           | Savings health indicator          |
   | `monthly_expense_burden_ratio` | (expenses + emi) / income  | Spending pattern                  |
   | `emergency_fund_months`        | savings / (expenses + emi) | Safety buffer in months           |

2. **Product quality features** (`review_features.py`)

   Input: `data/processed/review_preprocessed.jsonl`
   Output: `data/features/product_featured.jsonl` (one row per product)

   | Feature           | Formula                 | Purpose                 |
   | ----------------- | ----------------------- | ----------------------- |
   | `rating_variance` | std(rating) per product | Rating consensus signal |

   > **Note:** `average_rating` and `rating_number` (equivalent to `num_reviews`) already exist in the product metadata. The only feature requiring individual review data is `rating_variance`, which measures how polarized opinions are about a product.

3. **Handle edge cases**
   - Zero income: ratios set to NaN (XGBoost handles natively)
   - Division by zero: safe handling with NaN defaults
   - Single-review products: rating_variance defaults to 0.0

4. **Outputs**
   - `data/features/financial_featured.csv` — Financial profiles enriched with health metrics
   - `data/features/product_featured.jsonl` — Product-level features
   - `data/features/review_featured.jsonl` — Review-level features

---

## Phase 12: Feature Validation

### Objective

Validate that engineered features are correctly calculated and within expected ranges.

### Steps

1. **Define feature expectations**

   **Financial Features:**
   | Feature | Expected Range | Validation |
   |---------|---------------|------------|
   | `discretionary_income` | Can be negative | Check not NaN where income > 0 |
   | `debt_to_income_ratio` | 0 to ~2 | Flag if > 2; NaN only when income = 0 |
   | `savings_to_income_ratio` | 0 to ~1 | Flag if > 1; NaN only when income = 0 |
   | `monthly_expense_burden_ratio` | 0 to ~1 | Flag if > 1 |
   | `emergency_fund_months` | >= 0 | NaN only when obligations = 0 |

   **Product Quality Features:**
   | Feature | Expected Range | Validation |
   |---------|---------------|------------|
   | `rating_variance` | >= 0 | Check not negative; 0.0 for single-review products |

2. **Run validation**
   - No unexpected NaN or Inf values
   - Ratios within reasonable bounds
   - All expected features present
   - Product count in variance output matches product count in reviews

3. **Cross-validate calculations**
   - Sample records: manually verify formulas
   - Edge cases: zero income, single review products

4. **Handle failures**
   - Log calculation errors
   - Identify problematic source records


---

## Phase 13: Version Features (DVC Checkpoint #3)

### Objective

Version control the feature-engineered data for reproducibility.

### Steps

1. **Add features to DVC** (run from `data_pipeline/dags/data`)

   ```bash
   cd data_pipeline/dags/data
   dvc add features
   ```

   This creates/updates `features.dvc`.

2. **Commit and push**

   ```bash
   git add features.dvc
   git commit -m "Add features v1.0"
   dvc push
   ```

3. **Tag the version** (optional)
   ```bash
   git tag -a "data-features-v1.0" -m "Feature-engineered data"
   ```

### Why Version Here?

- Ties financial health features to model training versions
- Enables comparison of feature distributions across pipeline runs
- Experiment reproducibility

---

## Phase 14: Load to Database

### Objective

Load processed data, engineered features, and product embeddings into PostgreSQL (relational) and pgvector (vector) databases for the SavVio application.

### Environment Configuration

| Environment | Database                           | Connection      |
| ----------- | ---------------------------------- | --------------- |
| Development | Local PostgreSQL + pgvector        | localhost:5432  |
| Production  | GCP Cloud SQL + pgvector extension | Cloud SQL proxy |

### Steps

1. **Identify datasets to load**
   - `data/features/financial_featured.csv` — Financial profiles with health metrics
   - `data/processed/product_preprocessed.jsonl` — Product catalog
   - `data/features/product_featured.jsonl` — Rating variance per product (merged onto products during load)
   - `data/processed/review_preprocessed.jsonl` — Individual reviews

2. **Define database schema (tables)**

   - PostgreSQL Tables: Schema defined in `src/database/db_schema.py`

   - pgvector Table: For embeddings - defined in `src/database/vector_loader.py`

3. **Implement data loaders**

   **postgres_loader.py:**
   - `load_financial_profiles(df, engine)` — Load financial profiles with features
   - `load_products(products_path, rating_variance_path, engine)` — Load products with rating_variance merged
   - `load_reviews(df, engine)` — Load individual reviews
   - `get_engine(env)` — Get connection based on environment

   **vector_loader.py:**
   - `generate_embeddings(texts, model)` — Create embeddings from product descriptions
   - `load_embeddings(product_ids, embeddings, engine)` — Store in pgvector

4. **Environment-based configuration**

   ```yaml
   # config/database.yaml
   development:
     host: localhost
     port: 5432
     database: savvio_dev
     user: ${DB_USER}
     password: ${DB_PASSWORD}

   production:
     host: /cloudsql/project:region:instance
     port: 5432
     database: savvio_prod
     user: ${DB_USER}
     password: ${DB_PASSWORD}
   ```

5. **Load data**
   - Truncate existing data (or upsert strategy)
   - Load financial profiles (with pre-computed features)
   - Load products (with rating_variance merged)
   - Load reviews
   - Generate and load embeddings



---

## Phase 15: Bias Detection & Mitigation

### Objective

Detect and mitigate data representation bias across financial profiles and product/review data. The goal is to ensure the pipeline produces balanced, representative data so downstream models and the decision engine don't systematically disadvantage any subgroup.

### Why Two Separate Tracks?

Each data track is analyzed independently because they represent fundamentally different populations (users vs. products) and carry different bias risks:

| Track          | Population         | Bias Risk                                                                                         |
| -------------- | ------------------ | ------------------------------------------------------------------------------------------------- |
| Financial      | User profiles      | Underrepresentation of financially vulnerable users — the people SavVio is designed to help most  |
| Product/Review | Products & reviews | Skewed category coverage, price range gaps, or unreliable quality signals for low-review products |

> **Note:** Decision outcome bias (e.g., does the Green/Yellow/Red recommendation system unfairly penalize low-income users?) is tested in Phase 3: Model Development, once the Deterministic Financial Logic Engine and affordability calculations exist. The data pipeline focuses on **data representation bias** only.

### Steps

1. **Financial data — Slice analysis**

   | Slice Dimension       | Groups                                    | What to Check                           |
   | --------------------- | ----------------------------------------- | --------------------------------------- |
   | Income bracket        | Low (<$3k), Medium ($3k-$7k), High (>$7k) | Sufficient low-income representation?   |
   | Debt-to-income ratio  | Low (<0.2), Medium (0.2-0.4), High (>0.4) | Balanced coverage across debt levels?   |
   | Expense burden        | Low (<0.5), Medium (0.5-0.8), High (>0.8) | Are high-burden users underrepresented? |
   | Emergency fund months | Critical (<1), Low (1-3), Healthy (3+)    | Enough financially stressed profiles?   |

   **Analysis:**
   - Count records per slice — flag slices with <10% of total records
   - Compare feature distributions across slices (mean, median, std of each financial metric)
   - Identify if any slice has significantly different feature distributions that could bias model training

   **Why this matters for SavVio:**
   If the financial data skews toward high-income, healthy profiles, the model won't learn effective decision boundaries for users who are financially vulnerable. These are the users most likely to benefit from a "Red Light" recommendation, and the system must work well for them.

2. **Product/review data — Slice analysis**

   | Slice Dimension               | Groups                                         | What to Check                                         |
   | ----------------------------- | ---------------------------------------------- | ----------------------------------------------------- |
   | Product category              | Electronics, Clothing, Home, etc.              | Any category with <5% of products?                    |
   | Price range                   | Budget (<$25), Mid ($25-$200), Premium (>$200) | Balanced price representation?                        |
   | Average rating                | Low (<3), Medium (3-4), High (>4)              | Are low-rated products underrepresented?              |
   | Review volume (rating_number) | Few (<10), Some (10-100), Many (>100)          | Do low-review products lack reliable quality signals? |
   | Rating variance               | Low (<0.5), Medium (0.5-1.0), High (>1.0)      | Are polarizing products represented?                  |

   **Analysis:**
   - Count products per slice — flag underrepresented groups
   - Check if `rating_variance` is meaningful for low-review products (variance from 2 reviews is unreliable)
   - Verify price distribution covers the range users are likely to query about

   **Why this matters for SavVio:**
   If the product data is dominated by one category (e.g., electronics), the RAG retrieval and quality signals will perform poorly for other categories. If budget products are underrepresented, the system may lack good alternatives to recommend when giving a "Yellow Light."

3. **Evaluate for bias**
   - Are any critical slices underrepresented (<10% of records)?
   - Would the data gaps cause the system to perform worse for specific user groups?
   - Are there product categories where quality signals (rating_variance, avg_rating) are unreliable due to low review counts?

4. **Implement mitigation if needed**

   **Financial data:**
   - Oversample underrepresented income brackets or debt levels
   - Generate synthetic profiles for underrepresented slices
   - Document which slices are underrepresented and the expected impact

   **Product/review data:**
   - Flag products with fewer than N reviews as having low-confidence quality signals
   - Ensure category distribution covers common purchase types
   - Document category gaps and their impact on recommendations

5. **Document analysis**
   - The following subsection summarizes slices used, bias found, and mitigation strategies (aligned with the team’s Bias Detection document). Raw datasets are not modified; mitigation is applied at **model training time only**.

---

### Bias Detection Report (Phase 15 Summary)

This section describes the **slices used**, **bias found**, and **mitigation strategies** from representation bias analysis across Financial (17 columns), Product (10 columns), and Review (8 columns) datasets. The objective is to identify underrepresented high-risk financial groups and high-uncertainty product/review slices that could skew downstream affordability and recommendation models.

#### Slices Used for Bias Detection

**Financial (domain-informed risk bands)**

| Dimension                                                        | Bands / Logic                                                   | Threshold (flagged if) |
| ---------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------- |
| Discretionary income                                             | Negative (&lt;0), Tight (0–1000), Comfortable (&gt;1000)        | —                      |
| Debt-to-income (DTI)                                             | Safe (&lt;0.2), Warning (0.2–0.4), Risky (&gt;0.4)              | Warning &lt;10%        |
| Savings-to-income                                                | Fragile (&lt;0.25), Moderate (0.25–1.0), Strong (&gt;1.0)       | Fragile &lt;10%        |
| Monthly expense burden                                           | Comfortable (&lt;0.5), Tight (0.5–0.8), Overstretched (&gt;0.8) | —                      |
| Emergency fund months                                            | Quantile bands (Q1–Q4) + outliers                               | —                      |
| Income / expenses / loan amount / interest / term / credit score | Low, Medium, High (quantiles or domain bins)                    | High-risk band &lt;10% |
| Employment status                                                | Employed, Self-employed, Unemployed, Student                    | Category &lt;10%       |
| Region                                                           | Geographic categories                                           | —                      |
| user_id                                                          | Uniqueness check                                                | Uniqueness &lt;95%     |
| savings_balance                                                  | Near-zero, Low, Moderate, High                                  | Near-zero &lt;10%      |

**Product (uncertainty and coverage bands)**

| Dimension              | Bands / Logic                                          | Threshold (flagged if) |
| ---------------------- | ------------------------------------------------------ | ---------------------- |
| Price                  | Budget, Mid-range, Premium                             | —                      |
| Average rating         | Low, Medium, High                                      | —                      |
| rating_number          | Low / Medium / High confidence (review count)          | —                      |
| rating_variance        | Consensus, Mixed, Polarized, Single-review proxy (0.0) | —                      |
| Description / features | Length or count bands (0, 1–2, 3–5, 6+)                | —                      |
| details (Brand)        | Rare-brand detection                                   | Rare brands &lt;5%     |
| category               | Long-tail category coverage                            | Category &lt;5%        |
| product_id             | Uniqueness                                             | —                      |

**Review (signal and coverage bands)**

| Dimension                  | Bands / Logic                  | Threshold (flagged if)     |
| -------------------------- | ------------------------------ | -------------------------- |
| rating                     | Negative, Neutral, Positive    | Neutral or minority &lt;5% |
| verified_purchase          | True, False                    | Minority class &lt;5%      |
| helpful_vote               | None, Low, Medium, High        | High &lt;5%                |
| review_title / review_text | Short, Medium, Long, Empty     | —                          |
| user_id                    | Uniqueness                     | Uniqueness &lt;95%         |
| Reviews per product        | 1, 2–5, 6–20, 21+ (cold-start) | —                          |

---

#### Bias Found (Phase 15 Outputs)

**Financial (flagged)**

| Column / slice              | Finding                                                                 | Risk                                                                                    |
| --------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **savings_balance**         | Near-zero savings severely underrepresented (~0%)                       | Financial vulnerability under-captured; model may miss users most in need of Red Light. |
| **employment_status**       | Unemployed (9.93%) and Student (9.91%) slightly below 10%               | Financially vulnerable groups underrepresented.                                         |
| **debt_to_income_ratio**    | Warning band (0.2–0.4) = 3.48%                                          | Mid-risk users underrepresented; model may learn binary Safe vs Risky.                  |
| **savings_to_income_ratio** | Fragile (&lt;0.25) ~1.5%                                                | Long-term vulnerability underrepresented.                                               |
| **emergency_fund_months**   | Critical (&lt;1 month) and Fragile (1–3 months) highly underrepresented | Emergency-risk users rare; classifier may rarely predict Red in real distress cases.    |

**Review (flagged)**

| Column / slice        | Finding         | Risk                                   |
| --------------------- | --------------- | -------------------------------------- |
| **user_id**           | Uniqueness ~83% | Repeat reviewers may dominate signals. |
| **rating**            | Neutral = 4.89% | Middle sentiment underrepresented.     |
| **verified_purchase** | False ~4.16%    | Non-verified reviews underrepresented. |
| **helpful_vote**      | High = 1.5%     | High-signal reviews scarce.            |

**Product (flagged)**

| Column / slice              | Finding                                                                   | Risk                                                    |
| --------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------- |
| **category**                | Many long-tail categories &lt;5%                                          | Coverage skew toward popular categories; category bias. |
| **details (Brand)**         | Many rare brands &lt;5%                                                   | Model may overfit to dominant brands.                   |
| **Low-confidence products** | High share of single-review / low-count items (e.g. rating_variance == 0) | Cold-start risk; uncertainty at serving time.           |

---

#### Mitigation Strategies (Training-Time Only)

Mitigation is applied **only at model training time**. The raw dataset is not modified.

**Financial**

- **DTI Warning band:** Oversample Warning (0.2–0.4) during training; ensure moderate-debt users are sufficiently represented in the training split.
- **Savings-to-income (Fragile):** Oversample Fragile users; ensure minimum exposure of low-savings profiles; if needed, controlled synthetic low-savings profiles (clearly labeled).
- **Emergency fund (Critical / Fragile):** Oversample Critical and Fragile runway users; stress-test classifier on &lt;3 month runway users; optionally controlled synthetic emergency profiles. **Highest priority mitigation.**
- **Employment (Unemployed / Student):** Stratified sampling by employment_status so vulnerable groups are represented.

**Review**

- **Rating (Neutral):** Stratified sampling by sentiment class (Negative / Neutral / Positive).
- **verified_purchase (False):** Oversample non-verified reviews during training.
- **helpful_vote (High):** Weight high-helpfulness reviews more during training.
- **user_id (repeat reviewers):** User-level deduplication or per-user weighting to avoid repeat reviewers dominating.

**Product**

- **Category (long-tail):** Stratified sampling or category grouping to reduce bias toward popular categories.
- **Brand (rare):** Group rare brands or apply brand smoothing to avoid overfitting to major brands.
- **Low-confidence (single-review):** Flag as low-confidence at serving time; down-weight in recommendation logic; optionally require minimum review count.

**Cross-cutting**

- Stratified sampling across all flagged slices where applicable.
- Controlled oversampling of underrepresented high-risk slices (with caps to avoid distortion).
- Evaluation stress tests on vulnerable slices (low savings, DTI Warning, cold-start products).
- No synthetic data in raw pipeline unless model performance on vulnerable slices remains poor after oversampling.

---

#### Trade-offs and Design Rationale

| Choice                               | Rationale                                                                                                |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| **No raw data mutation**             | Preserves real-world distributions and avoids synthetic artifacts in source-of-truth datasets.           |
| **Mitigation at training time only** | Keeps the data pipeline deterministic and auditable; fairness controls are applied where models are fit. |
| **Oversampling with caps**           | Reduces underrepresentation while limiting overfitting to rare slices; validated via cross-validation.   |
| **Long-tail / category grouping**    | Improves fairness across categories but may increase variance; addressed with grouping and smoothing.    |

**Limitations and next steps**

- Incorporate fairness-aware sampling directly into model training pipelines (Phase 3).
- Add per-slice performance metrics (e.g. recall on vulnerable users).
- Consider post-training calibration and threshold tuning for high-risk (Red) decisions.

---



### Future: Decision Outcome Bias (In Model Development)

Once the Deterministic Financial Logic Engine is built, a separate bias analysis should test:

- Do Green/Yellow/Red recommendations distribute fairly across income brackets?
- Does the affordability score systematically penalize certain financial profiles?
- Are certain product categories more likely to receive Red Light recommendations regardless of user finances?

This analysis requires the full decision pipeline and belongs in the model development phase, not the data pipeline.

---

## Phase 16: Pipeline Orchestration (Airflow DAGs)

### Objective

Structure the entire pipeline using Apache Airflow DAGs with conditional branching to manage complex error-handling logic, parallel processing, and dependencies.

### Implementation Setup

1. **Architecture**
   - We run a containerized Airflow environment via `docker-compose.yaml` (including PostgreSQL and Redis backends).
   - Core DAG file: `dags/data_pipeline_airflow.py`

2. **Implemented DAG Structure**
   The DAG achieves maximum concurrency while respecting data dependencies and executing specific validation branches to catch failures. Here is the implemented structure:

   ```
   [ingest_financial] ───────┐
   [ingest_product]   ───────┼──> [check_ingestion] ──(failed?)──> [email_error_at_ingestion]
   [ingest_review]    ───────┘          │ (success)
                                        ▼
   [validate_raw_financial] ─┐
   [validate_raw_product]   ─┼──> [check_raw_validation] ──(failed?)──> [email_error_error_raw_validation]
   [validate_raw_review]    ─┘          │
                                        ▼
   [detect_anomalies_*]     ────> [check_anomalies] ──(failed?)──> [email_error_at_anomalies]
                                        │
                                        ▼
   [preprocess_financial] ───       (parallel)
   [preprocess_product]   ───> [preprocess_review]
                                        │
                                        ▼
   [check_preprocessing] ──(failed?)──> [email_error_at_preprocessing]
                                        │
                                        ▼
   [validate_processed_*] ──> [check_processed_validation] ──(failed?)──> [email_error_at_processed_validation]
                                        │
                                        ▼
   [feature_financial]    ───┐
   [feature_reviews]      ───┴──> [check_feature_engineering] ──(failed?)──> [email_error_at_feature_engineering]
                                        │
                                        ▼
   [validate_featured]    ──────> [check_featured_validation] ──(failed?)──> [email_error_at_featured_validation]
                                        │
                                        ▼
   [setup_database] ──> [load_financial, load_product] ──> [load_review] ──> [generate_load_embeddings]
                                        │
                                        ▼
   [check_db_loading] ──(failed?)──> [email_error_at_DB_loading]
                                        │(success)
                                        ▼
   [email_pipeline_success] ──> [pipeline_sentinel]
   ```

3. **Task Implementation**
   - The DAG tasks primarily map directly to `src/` modules using `PythonOperator`.
   - Error detection is implemented via `BranchPythonOperator` blocks (e.g., `make_branch_check(...)`) that evaluate upstream task states and trigger `EmailOperator` tasks on failure.

---

## Phase 17: Testing

### Objective

Provide comprehensive coverage of all data modules and tasks before pipeline deployment using parameterized test files and robust service mocks.

### Implementation

1. **Testing Standards Applied**
   - **Mocking Extraneous Services**: Stubs are aggressively used (e.g., `_stub` in `test_data_pipeline_airflow.py` and `_stub_google` in `test_gcs_loader.py`) to effectively isolate module tests from requiring active GCP or Airflow metadata connections.
   - **Format Standard:** All test files include module docstrings, section dividers, and structural comments.
   - **Bias Detection Validation:** The bias detection modules (`financial_bias.py`, `product_bias.py`, `review_bias.py`, `run_bias.py`) are exercised end-to-end by running `python3 src/bias/run_bias.py` against the engineered datasets. The logged outputs and the \"Bias Detection Report\" section above form the test oracle for Phase 15.

2. **Running the Tests**
   - `pytest <path_to_tests>/tests/ -v`

---

## Phase 18: Tracking, Logging & Monitoring

### Objective

Ensure data pipeline execution observability.

### Implementation

1. **Module-wide Logging**
   - `src/utils.py` contains shared `logging` config that formats logs by `[timestamp] [level] [module_name]`.
   - Every individual script explicitly initializes its logger instance globally using `logger = logging.getLogger(__name__)`.
2. **Airflow Alerts**
   - If any core task fails, branching conditions inside Airflow dynamically evaluate the task context state and trigger an `EmailOperator` (like `email_error_at_preprocessing`) notifying pipeline maintainers precisely which phase failed.

---

## Phase 19: Pipeline Optimization

### Objective

Diagnose runtime bottlenecks and pipeline faults.

### Implementation

1. **DAG Race Condition Mitigation**
   - _Problem:_ A race condition dropped merged data due to overlapping run times in `preprocess_product` and `preprocess_review`.
   - _Solution:_ Airflow dependencies were explicitly enforced for these two segments `preprocess_product >> preprocess_review` to ensure product mapping succeeds before dependency resolution.

2. **DVC (Data Version Control) Checkpoints**
   - Intermediary DVC markers allow the pipeline data to be cached at raw, processed, and featured states without repeating slow upstream execution times. Data commits are persisted in GCS.

3. **Orchestration Concurrency**
   - Tasks like `ingest_financial`, `ingest_product`, and `ingest_review` execute totally asynchronously without blocking execution threads since they exist independently of one another.
