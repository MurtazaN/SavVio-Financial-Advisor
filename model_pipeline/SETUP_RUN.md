## Runbook

### Option 1: Docker (recommended)

#### Prerequisites

Run these checks from the **repo root** before starting:

```bash
# 1) savviocore must exist — it is mounted as a volume into the training container.
ls savviocore

# 2) Training data must be present.
ls model_pipeline/data/training_scenarios.csv

# 3) .env must exist in model_pipeline/. Copy from the example if missing.
ls model_pipeline/.env || cp model_pipeline/.env.model.example model_pipeline/.env
```

#### Run

```bash
cd model_pipeline

# Build images, start all services, and run the pipeline automatically.
# Service startup order (enforced by health checks):
#   1. postgres  → MLflow metadata backend
#   2. storage   → RustFS S3-compatible artifact store
#   3. mlflow    → Tracking server at http://localhost:5000
#   4. ml-trainer → Runs the full pipeline, then exits
docker compose up --build
```

Pipeline progress streams live to the terminal:
```
savvio-ml-trainer  | Starting End-to-End ML Pipeline...
savvio-ml-trainer  | [1/6] Initializing runtime + MLflow...
savvio-ml-trainer  | [2/6] Preparing data (load/encode/split)...
savvio-ml-trainer  | [3/6] Training baseline candidates...
savvio-ml-trainer  | [4/6] Running hyperparameter tuning on best baseline...
savvio-ml-trainer  | [5/6] Selecting best model (F1 + bias gate)...
savvio-ml-trainer  | [6/6] Running final evaluation on held-out test set...
savvio-ml-trainer  | Pipeline Complete!
savvio-ml-trainer exited with code 0
```

`ml-trainer` exiting with code 0 confirms the pipeline ran successfully.
The other services (postgres, storage, mlflow) keep running so you can inspect results.

#### Check results

**MLflow UI** — runs, nested runs, metrics, artifacts, and registered models:
```
http://localhost:5000
```
Navigate to: **Experiments → savvio_model_pipeline** to see baseline + tuned runs.
Navigate to: **Models → SavvioChampionModel** to see the registered champion model.

**Local models** — champion model and label encoder are saved to the host machine:
```bash
ls model_pipeline/models/
```
These files are written by `save_best_model_local()` and survive container teardown via the volume mount `./models:/app/models`.

**Logs** — review the full training log after the fact:
```bash
docker compose logs ml-trainer
```

#### Tear down

```bash
# Stop all services (postgres, storage, mlflow). Volumes are preserved.
docker compose down

# Full reset — also wipes postgres and storage volumes (clears MLflow data):
docker compose down -v
```

### Option 2: Local virtualenv
```bash
# 1) Move into the model pipeline folder so relative paths resolve there.
cd model_pipeline

# 2) Create and activate a local environment for isolated dependencies.
python3.11 -m venv .venv
source .venv/bin/activate

# 3) Install pipeline dependencies (MLflow, XGBoost, LightGBM, Optuna, etc.).
pip install -r model-requirements.txt

# 4) Start MLflow tracking server in this folder (Terminal A).
#    - backend-store-uri: run metadata DB
#    - default-artifact-root: model/artifact files
mlflow server \
	--backend-store-uri sqlite:///mlflow.db \
	--default-artifact-root ./mlruns \
	--host 127.0.0.1 \
	--port 5000

# 5) In a second terminal (Terminal B), activate env again and run pipeline.
cd model_pipeline
source .venv/bin/activate

# 6) Point training script to local MLflow server and run end-to-end pipeline.
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python src/run_pipeline.py

# 7) Open MLflow UI to review runs, metrics, artifacts, and registered models.
open http://127.0.0.1:5000
```

### Individual Components
```bash
# Test DB connectivity (once DB integration is live)
python src/data/db_loader.py

# Run data validation after loading
python src/data/validate_data.py

# View MLflow UI
open http://localhost:5000
```

### Running Tests
```bash
pytest tests/ -v
```
