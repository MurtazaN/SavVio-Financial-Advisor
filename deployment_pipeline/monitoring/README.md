# SavVio Monitoring — Prometheus + Grafana

Observability stack for the SavVio API inference pipeline. Tracks API latency,
recommendation distribution (GREEN/YELLOW/RED), ML model confidence, guardrail
failures, and Layer 2 downgrades.

## Architecture

```
┌──────────────────────────────┐
│   SavVio FastAPI API (:3500) │
│                              │
│  prometheus-fastapi-         │
│  instrumentator              │
│  + custom business metrics   │
│         │                    │
│    GET /metrics              │
└─────────┬────────────────────┘
          │
    ┌─────┴──────┐
    │            │
    ▼            ▼
┌────────┐  ┌─────────────┐
│  Local │  │ Grafana Cloud│
│  Prom  │  │ (remote-write│
│ (:9090)│  │   push)      │
└───┬────┘  └──────┬──────┘
    │              │
    ▼              ▼
┌────────┐  ┌─────────────┐
│ Local  │  │   Cloud     │
│Grafana │  │ Dashboards  │
│(:3001) │  │             │
└────────┘  └─────────────┘
```

---

## Quick Start (Local Development)

### Prerequisites
- Docker & Docker Compose installed
- SavVio API running on port 3500

### 1. Start the Monitoring Stack

```bash
cd deployment_pipeline/monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

### 2. Start the SavVio API

```bash
# From the project root directory (/SavVio)
./deployment_pipeline/run.sh api
```

### 3. Access Dashboards

| Service    | URL                       | Credentials     |
|------------|---------------------------|-----------------|
| Grafana    | http://localhost:3001      | admin / savvio  |
| Prometheus | http://localhost:9090      | —               |
| API Metrics| http://localhost:3500/metrics | —            |

### 4. Generate Sample Data

Hit the API a few times to populate the dashboards:

```bash
# Health check
curl http://localhost:3500/health

# Inference request (generates GREEN/YELLOW/RED metrics)
curl -X POST http://localhost:3500/predict \
  -H "Content-Type: application/json" \
  -d '{"user_query": "Can I buy the Sony WH-1000XM5?", "user_id": "user_001"}'
```

### 5. Stop the Stack

```bash
docker-compose -f docker-compose.monitoring.yml down
# Add -v to also remove persistent volume data:
# docker-compose -f docker-compose.monitoring.yml down -v
```

---

## Custom Metrics Reference

All metrics use the `savvio_` prefix.

### Inference Pipeline

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `savvio_inference_total` | Counter | `recommendation`, `evaluation_mode` | Total inference requests |
| `savvio_inference_duration_seconds` | Histogram | `recommendation`, `evaluation_mode` | E2E pipeline latency |
| `savvio_active_requests` | Gauge | — | In-flight requests |

### ML & Guardrails

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `savvio_ml_confidence` | Histogram | — | ML confidence score distribution |
| `savvio_guardrail_failures_total` | Counter | — | LLM guardrail check failures |
| `savvio_layer2_downgrades_total` | Counter | `from_color`, `to_color` | Layer 2 quality downgrades |

### LLM Provider

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `savvio_llm_latency_seconds` | Histogram | `provider`, `operation` | LLM response time |

### HTTP (auto-instrumented)

| Metric | Type | Description |
|--------|------|-------------|
| `http_request_duration_seconds` | Histogram | Per-endpoint HTTP latency |
| `http_requests_total` | Counter | Total HTTP requests by status |
| `savvio_http_requests_inprogress` | Gauge | In-progress HTTP requests |

---

## Grafana Cloud Setup (Production)

### 1. Create a Grafana Cloud Account

Sign up at [grafana.com/cloud](https://grafana.com/cloud) — the free tier is
sufficient for SavVio.

### 2. Get Your Remote Write Credentials

In Grafana Cloud:
1. Go to **Connections → Add new connection → Hosted Prometheus metrics**
2. In the "1. Choose a method for forwarding metrics" section, select the **"From my local Prometheus server"** card.
3. Scroll down to the "3. Set the configuration" section.
4. Under "Use an API token", enter a Token name and click **"Create token"**.
5. A `remote_write` YAML block will be generated. Note the 3 values from it:
   - `url` (your endpoint)
   - `username` (your numeric instance ID)
   - `password` (your new API key)

### 3. Configure Environment Variables

Add to your `.env` or Cloud Run environment:

```bash
GRAFANA_CLOUD_REMOTE_WRITE_URL=https://prometheus-prod-XX-prod.grafana.net/api/prom/push
GRAFANA_CLOUD_USERNAME=123456
GRAFANA_CLOUD_API_KEY=glc_xxx...
```

### 4. Import the Dashboard

In Grafana Cloud:
1. Go to **Dashboards → Import**
2. Upload `provisioning/dashboards/savvio-monitoring.json`
3. Select your Prometheus data source
4. The dashboard will auto-populate with your metrics

---

## Configuration

All monitoring settings are in `deployment_pipeline/api/config.py`:

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `METRICS_ENABLED` | `true` | Enable/disable all Prometheus metrics |
| `METRICS_PREFIX` | `savvio` | Prefix for custom metric names |
| `GRAFANA_CLOUD_REMOTE_WRITE_URL` | `""` | Grafana Cloud push endpoint |
| `GRAFANA_CLOUD_USERNAME` | `""` | Grafana Cloud instance ID |
| `GRAFANA_CLOUD_API_KEY` | `""` | Grafana Cloud API key |

---

## Disabling Metrics

Set `METRICS_ENABLED=false` in your environment to completely disable metrics
collection. The `/metrics` endpoint will not be exposed and no counters/histograms
will be incremented.
