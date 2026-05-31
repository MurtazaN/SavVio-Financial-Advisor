# SavVio Phase 3: Backend API & Inference Layer

This directory contains the production-ready FastAPI backend for SavVio. It acts as the thin orchestration layer that chains together the deterministic rules engine, the ML model predictions, and the LLM response mechanisms.

By the time the request finishes, the user gets an authoritative Green/Yellow/Red recommendation, along with a cohesive explanation of *why* they were recommended that option.

---

## 1. Prerequisites

You must use the `model_pipeline` Python virtual environment for all API dependencies. It shares the same database schemas, configurations, and core features. 

Ensure you have your environment set to DEV in `.env` (it defaults to `dev` if absent) so you hit the right database instance. You also need local mock/model dependencies.

If you don't already have the dependencies installed:
```bash
# From project root
source model_pipeline/.venv/bin/activate
pip install -r model_pipeline/model-requirements.txt
pip install -r deployment_pipeline/requirements.txt
```

---

## 2. API Architecture

The backend follows a **dependency-injected, stage-decomposed** architecture:

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `main.py` | FastAPI app, endpoints, middleware. Endpoints receive resources via `Depends()`. |
| `dependencies.py` | FastAPI dependency providers (`get_model_manager`, `require_db`). Enables clean test overrides. |
| `inference.py` | Inference orchestrator, decomposed into 7 pipeline stages (see below). |
| `model_loader.py` | Resource lifecycle — focused loader classes composed by `ModelManager`. |
| `schemas.py` | Pydantic request/response models. |
| `products_catalog.py` | Product filtering and pagination for catalog browse. |
| `quality_signals.py` | Maps Layer 2 features into API-friendly signal views. |
| `config.py` | Centralized environment configuration. |

### Inference Pipeline Stages

`inference.py` decomposes the prediction into independently testable stages:

```
run_inference(request, manager)
  │
  ├─ Stage 1: _load_user_financial_profile()  → DB lookup
  ├─ Stage 2: _resolve_product()              → intent parsing / product match
  ├─ Stage 3: _compute_financial_features()   → affordability + feature guards
  ├─ Stage 4: _run_layer1_engine()            → deterministic GREEN/YELLOW/RED
  ├─ Stage 5: _run_layer2_engine()            → downgrade via product/review signals
  ├─ Stage 6: _score_ml_model()               → XGBoost confidence (informational)
  ├─ Stage 7: _generate_explanation()          → LLM response + guardrails
  │
  └─ _build_response()                        → assemble PredictResponse
```

Each stage returns a typed dataclass (`ProductResolution`, `FinancialResult`, `L1Result`, `L2Result`, `MLScore`) so inputs and outputs are explicit.

### Resource Loaders (model_loader.py)

`ModelManager` is a composition root that delegates to focused classes:

| Class | Owns |
|-------|------|
| `ModelArtifactLoader` | ML model (MLflow pyfunc + XGBoost fallback) |
| `LabelEncoderLoader` | sklearn LabelEncoder |
| `FeaturePipelineLoader` | sklearn preprocessing pipeline |
| `DatabaseManager` | SQLAlchemy engine lifecycle |
| `LLMManager` | LLM provider (Groq / Gemini / mock) |

### Dependency Injection (dependencies.py)

Endpoints declare what they need via FastAPI's `Depends()`:

```python
@app.post("/predict")
async def predict(
    request: PredictRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    return run_inference(request, manager)
```

Tests override dependencies cleanly — no monkeypatching:

```python
app.dependency_overrides[get_model_manager] = lambda: mock_manager
```

### Request Flow

1. **`main.py`** intercepts `/predict` and validates inputs (Pydantic models in `schemas.py`).
2. **`inference.py`** is the orchestrator: 
   - Uses pgvector (`sentence-transformers`) to resolve the user's natural language to a Database `product_id`.
   - Fetches the user's financial profile from the database (`user_id`).
   - Runs **Layer 1** (Deterministic Affordability Engine).
   - Runs **Layer 2** (Downgrade Engine based on Product/Review ML Features).
   - Generates confidence scores via XGBoost (`ModelManager`).
     - Confidence is derived from class probabilities (`predict_proba`).
     - The loader uses the MLflow `pyfunc` model for `predict`, but it also reloads the native XGBoost/LightGBM
       flavor to reliably obtain `predict_proba`.
     - If the artifact cannot provide `predict_proba`, the API will still respond with a recommendation but
       `confidence` may be `null` (check server logs for the scoring reason).
   - Requests a response generation block from the LLM based on specific inputs.
   - Triggers LLM Guardrails to ensure output safety.
3. **`model_loader.py`** manages resource lifecycle (the Database Engine, the XGBoost MLflow artifact, the Label Encoder, and Precomputed category statistics) to stay loaded in memory for fast performance.

### Catalog browse, reviews, and hypothetical mode

- **`GET /products`** — Lists rows from `products` for catalog selection in the UI. Supports `q` (substring on `product_name`), `price_min`, `price_max`, `limit` (default from `PRODUCT_BROWSE_DEFAULT_LIMIT`), and `offset`. If `price_min` / `price_max` are omitted, the API uses **`PRODUCT_BROWSE_PRICE_MIN`** and **`PRODUCT_BROWSE_PRICE_MAX`** from the environment (see `deployment_pipeline/api/config.py`).
- **Product reviews and Layer 2 (downgrade engine)** load only when inference has a resolved **`product_id`** that exists in `products` (e.g. `POST /predict` with `product_id`, or a successful catalog match from natural language). If there is **no** catalog match and the engine uses **stated price only** (`evaluation_mode: hypothetical`), review-derived features are not applied for that request.

---

## 3. Running the API Locally

Run Uvicorn from the **SavVio repository root** (the folder that contains `deployment_pipeline/`, `model_pipeline/`, and `savviocore/`). The inference layer imports packages from `model_pipeline/src` and `savviocore/src`, so you must set `PYTHONPATH`:

```bash
cd /path/to/SavVio

# Option A: run script (handles PYTHONPATH automatically)
./deployment_pipeline/run.sh api

# Option B: manual
export PYTHONPATH="model_pipeline/src:savviocore/src:."
uvicorn deployment_pipeline.api.main:app --host 0.0.0.0 --port 3500 --reload
```

If you run from another directory (for example only `model_pipeline/src`) without this `PYTHONPATH`, you will get `ModuleNotFoundError: No module named 'deployment_pipeline'` or import errors for `llm` / `savviocore`.

**Note:** `model_pipeline/src/data/db_loader.py` is used only by the **training pipeline** (`run_pipeline.py`), not by this API. Changing or reverting it does not start or stop the backend.

The server initializes everything and will typically output:
```log
[INFO] Loading model manager resources...
[INFO] Loaded model via mlflow.pyfunc from: .../model_pipeline/models/artifacts
[INFO] Native LightGBM model available for predict_proba (.../model_pipeline/models/artifacts)
[INFO] Database connected (env=dev)
[INFO] LLM provider initialized: mock
[INFO] Model manager fully loaded.
```

If it fails to connect to the DB or load the model, it will still start but operate in a gracefully degraded state.

---

## 4. Testing Endpoints

Once the Uvicorn server is running locally on port `3500`, you can test the three primary REST endpoints from another terminal.

*(Note: We use `U01157` as a sample User ID for local tests. You can query your local `financial_profiles` table for others).*

### Check API Health
Validates that connections to the Model, DB, and LLM are fully loaded and operational.
```bash
curl -s http://localhost:3500/health
```

### User financial profile (dashboard)
Returns the `financial_profiles` row for the React dashboard:
```bash
curl -s http://localhost:3500/user/U01157/profile
```

### Browse products (catalog picker)
```bash
curl -s "http://localhost:3500/products?limit=20&q=headphone"
```

### Direct Product Evaluation
If you already know the specific `product_id`, skip Natural Language parsing and evaluate directly:
```bash
curl -s "http://localhost:3500/user/U01157/evaluate?product_id=B07NPZ8YB1" 
```

### Natural Language Search & Evaluation
Pass raw natural language inputs. The engine will parse the intent, find the nearest `product_id` in pgvector, and evaluate:
```bash
curl -s -X POST http://localhost:3500/predict \
  -H "Content-Type: application/json" \
  -d '{
      "user_query": "Can I sensibly afford to buy a gas cooktop today?", 
      "user_id": "U01157"
  }'
```

### Example Output Response
```json
{
    "recommendation": "YELLOW",
    "confidence": 0.85,
    "explanation": "I'd suggest thinking carefully before buying 22″x20″ Built in Gas Cooktop 4 Burners Stainless Steel Stove NG/L-P-G Gas Hob Cooktop Kitchen Built-In Cook Stove Easy to Clean (22″x20″) at $167.49. Your emergency fund could use some building up before taking on additional expenses. While you can technically afford this, it would be worth considering whether this is a need or a want right now.",
    "product_name": "22″x20″ Built in Gas Cooktop 4 Burners Stainless Steel...",
    "product_price": 167.49,
    "triggered_rules": [
        "yellow:low_resilience"
    ],
    "was_downgraded": false,
    "guardrail_passed": true
}
```

---

## 5. Running the Unit Tests

We have complete test coverage over the core deterministic engine (`test_decision_logic.py`), the request orchestration logic (`test_inference.py`), and the FastAPI definitions (`test_api.py`).

Tests use FastAPI's `dependency_overrides` for clean mocking — no monkeypatching required:

```python
# conftest.py sets up DI overrides automatically
app.dependency_overrides[get_model_manager] = lambda: mock_manager
```

**Run the suite (38 total tests ensures total coverage across rule logic & configurations):**

```bash
# Run tests from the project root
python -m pytest deployment_pipeline/tests/ -v --tb=short
```

---

## Next Steps for the Team
Now that Phase 3 has verified the orchestration correctly combines deterministic outputs with our generative templates, the next step (Phase 4) is locking down the deployment in a Docker Container, deploying to Google Cloud Run, and pointing it directly to the Cloud SQL artifacts.
