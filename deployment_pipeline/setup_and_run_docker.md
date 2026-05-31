# SavVio — Docker & CI/CD Setup Guide

How to build, run, and deploy the SavVio API and frontend using Docker.

---

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | v2+ (bundled with Docker Desktop) | `docker compose version` |
| Git | Any | `git --version` |
| Postgres | <15 | `postgres --version` |

localhost:5432 (user: `savvio`, password: `savvio_local`, db: `savvio_dev`) |

For GCP deployment only:
| Requirement | Version | Check |
|-------------|---------|-------|
| gcloud CLI | Latest | `gcloud --version` |
| GitHub repo access | Push to `main` | — |

---

## Option 1: Local Full-Stack (docker-compose)

Starts API + frontend + Postgres with zero cloud dependencies.

### Start everything

```bash
docker compose up --build
```

### Endpoints

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8501 |
| API | http://localhost:8080 |
| API docs (Swagger) | http://localhost:8080/docs |

### Verify

```bash
# API health check
curl -s http://localhost:8080/health | python3 -m json.tool

```

### Stop and clean up

```bash
# Stop services
docker compose down

# Stop and remove database volume (full reset)
docker compose down -v
```

### Optional: enable LLM explanations

Pass your LLM Model API key as an environment variable:

```bash
<LLM>_API_KEY=your-key-here docker compose up --build
```

---

## Option 2: Deploy to GCP via CI/CD

### One-time setup

1. **GitHub Secrets** — ensure these are set in your repo settings:

   | Secret | Purpose | Status |
   |--------|---------|--------|
   | `GCP_SA_KEY` | GCP service account JSON key | Should already exist |
   | `GCP_PROJECT_ID` | GCP project ID | Should already exist |
   | `API_URL_DEV` | Cloud Run API URL (e.g. `https://savvio-dev-api-xxx.run.app`) | **New — add this** |

2. **Terraform** — the Cloud Run services, Artifact Registry, and Cloud SQL must already be provisioned via `deployment_pipeline/terraform/`.

### Trigger a deployment

**Automatic:** Push to `main` with changes in any of these paths:
- `deployment_pipeline/api/**`
- `deployment_pipeline/frontend/src/**`
- `deployment_pipeline/requirements.txt`
- `savviocore/**`
- `model_pipeline/src/**`
- `model_pipeline/models/model/**`

**Manual:** GitHub Actions → `Deployment CI/CD` → Run workflow → select `dev` or `prod`.

### What the workflow does

```
test-api ──────────► build-push-api ──────┐
                                          ├──► deploy ──► notify
test-frontend ─────► build-push-frontend ──┘
```

1. **test-api** — `pytest deployment_pipeline/tests/`
2. **test-frontend** — `npm run lint` + `npm test`
3. **build-push-api** — Docker build → push to Artifact Registry (SHA-tagged)
4. **build-push-frontend** — Docker build with `VITE_API_BASE` → push to Artifact Registry
5. **deploy** — `gcloud run services update` for both API and frontend
6. **notify** — Success/failure notification

### Verify after deploy

```bash
# Get the deployed API URL
gcloud run services describe savvio-dev-api --region us-east1 --format='value(status.url)'

# Health check
curl -s https://savvio-dev-api-xxx.run.app/health | python3 -m json.tool
```

---

## Architecture Notes

### Model loading

The ML model is **bundled into the API Docker image** at build time. It is a self-contained MLflow pyfunc artifact at `model_pipeline/models/model/` For Local and For Production, it is stored in GCP **Artifact Registry** containing the classifier, feature pipeline, and label encoder.

When the model is retrained, rebuild the API image to pick up the new artifacts.

### Frontend build

Vite compiles the React app into static HTML/JS/CSS. The `VITE_API_BASE` environment variable is **baked into the JS bundle at compile time** (not configurable at runtime). This means:
- For local docker-compose: set to `http://localhost:8080`
- For GCP: set to the Cloud Run API URL via the `API_URL_DEV` GitHub secret

The static files are served by nginx in the production container.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `docker compose up` fails on frontend build | Ensure `deployment_pipeline/frontend/package-lock.json` exists. If missing: `cd deployment_pipeline/frontend && npm install --package-lock-only` |
| API starts but `db_connected: false` | Postgres container may still be starting. Wait for healthcheck (5-10s) or check `docker compose logs postgres` |
| Frontend loads but API calls fail | Check `VITE_API_BASE` build arg matches where the API is running |
| `model_loaded: false` in health check | Verify `model_pipeline/models/model/` contains `MLmodel`, `python_model.pkl`, and `artifacts/` |
| CI/CD build fails on push | Check GitHub Secrets are set (`GCP_SA_KEY`, `GCP_PROJECT_ID`, `API_URL_DEV`) |
| `npm ci` fails in Dockerfile | Regenerate lockfile: `cd deployment_pipeline/frontend && npm install --package-lock-only` |
| Image too large (>2GB) | Check `.dockerignore` is not missing — it excludes data files, experiments, and node_modules |
