# SavVio — Setup & Run Guide

Complete instructions to run the SavVio backend API and frontend locally.

---

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.10+ | `python --version` |
| Node.js | 18+ | `node --version` |
| PostgreSQL | Running locally on port 5432 | `pg_isready` |
| Git | Any | `git --version` |

The backend connects to a PostgreSQL database with preloaded `financial_profiles`, `products`, and `reviews` tables (see `savviocore` for schema and seeding).

---

## 1. Clone & Navigate

```bash
git clone https://github.com/nirajmehta960/SavVio.git
cd SavVio
```

---

## 2. Environment Variables

Copy the `.env` file at the project root. Key variables:

```bash
# Database (required)
DB_USER=postgres
DB_PASSWORD=*******
DB_HOST=localhost
DB_PORT=5432
DB_NAME=savvio_dp

# LLM provider (at least one required for non-mock mode)
GROQ_API_KEY=...          # Groq (llama-3.3-70b)
GEMINI_API_KEY=...        # Google Gemini (fallback)

# Environment
ENVIRONMENT=dev           # "dev" or "prod"
```

If no LLM key is set, the API starts with a **mock provider** (template-based responses instead of live LLM).

---

## 3. Run SavVio

The `run.sh` script can start both services in one terminal:

```bash
./deployment_pipeline/run.sh
```

This starts the backend (port 3500) and frontend (port 3000) as background processes. `Ctrl+C` stops both.

---

## Manual Backend Setup

### Install Python dependencies

The API shares the `model_pipeline` virtual environment:

```bash
cd model_pipeline
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r model-requirements.txt
pip install -r ../deployment_pipeline/requirements.txt
cd ..
```

### Verify model artifacts exist

The API loads the trained model from `model_pipeline/models/artifacts/`. If you haven't run the training pipeline yet, the API will start in **degraded mode** (no ML confidence scores, deterministic engine still works).

### Start the backend

```bash
# Option A: Using the run script
./deployment_pipeline/run.sh api

# Option B: Manual
export PYTHONPATH="model_pipeline/src:savviocore/src:."
uvicorn deployment_pipeline.api.main:app --host 0.0.0.0 --port 3500 --reload
```

Expected startup output:
```
[INFO] Loading model manager resources...
[INFO] Loaded model via mlflow.pyfunc from: .../model_pipeline/models/artifacts
[INFO] Database connected (env=dev)
[INFO] LLM provider initialized: groq
[INFO] Model manager fully loaded.
```

### Verify the backend

```bash
# Health check
curl -s http://localhost:3500/health | python -m json.tool

# User profile
curl -s http://localhost:3500/user/U01157/profile | python -m json.tool

# Product catalog
curl -s "http://localhost:3500/products?limit=5&q=headphone" | python -m json.tool

# Full inference
curl -s -X POST http://localhost:3500/predict \
  -H "Content-Type: application/json" \
  -d '{"user_query": "Can I afford a gas cooktop?", "user_id": "U01157"}'
```

---

## Manual Frontend Setup

```bash
cd deployment_pipeline/frontend
npm install
```

### Start the frontend

```bash
# Option A: Using the run script (from project root)
./deployment_pipeline/run.sh frontend

# Option B: Manual
cd deployment_pipeline/frontend
npm run dev
```

The app will be available at **http://localhost:3000**. API requests to `/api/*` are proxied to the backend at `:3500` automatically (configured in `vite.config.ts`).

---

## Running Tests

### Backend tests (pytest)

```bash
source model_pipeline/.venv/bin/activate
export PYTHONPATH="model_pipeline/src:savviocore/src:."
python -m pytest deployment_pipeline/tests/ -v --tb=short
```

### Frontend tests

```bash
cd deployment_pipeline/frontend

# Unit tests (Vitest)
npm run test

```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'deployment'` | Set `PYTHONPATH` as shown above, or use `run.sh` |
| `ModuleNotFoundError: No module named 'llm'` | Ensure `model_pipeline/src` is on `PYTHONPATH` |
| `Database initialization failed` | Check PostgreSQL is running and `.env` credentials are correct |
| `LLM provider initialization failed — using mock` | Set `GROQ_API_KEY` or `GEMINI_API_KEY` in `.env` |
| `Model artifact directory not found` | Run the training pipeline first, or accept degraded mode |
| Frontend shows blank page | Ensure backend is running on port 3500 |
