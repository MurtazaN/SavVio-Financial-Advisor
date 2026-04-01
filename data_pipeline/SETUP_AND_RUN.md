# SavVio Data Pipeline Setup & Execution Guide

This guide provides step-by-step instructions to **set up the environment and run the pipeline** so that it can be **reproduced on another machine** without errors. It supports the evaluation criteria for **Proper Documentation**, **Reproducibility**, and **Data Version Control (DVC)**.

**See also:** `README.md` for full pipeline design and phase-by-phase documentation.

---

## Prerequisites

- **Docker** and **Docker Compose** installed  
- **Minimum resources**: 6GB RAM (4GB allocated to Docker), 2 CPUs, ~10GB free disk  
  - **Recommended**: 8–10GB RAM (6GB allocated to Docker) and 4+ CPUs for smoother Airflow execution and DuckDB out-of-core merging
- **Git** (to clone the repo)
- Python 3.12+
- **PostgreSQL** reachable from Docker for pipeline outputs (**separate from Airflow's bundled Postgres**)  
  - Set via `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` in `.env`  
  - `pgvector` extension must be available/enabled (pipeline will attempt `CREATE EXTENSION IF NOT EXISTS vector`)
- A **Gmail account** with an App Password (for email notifications)
- **Google Cloud SDK** (optional; for direct GCS access)

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/nirajmehta960/SavVio.git
cd SavVio
```

### 2. Set Up Environment

Create the required directories and set up permissions:

```bash
mkdir -p data_pipeline/logs data_pipeline/plugins
```

> **Note:** Do not use `echo` commands to append variables to your `.env` file. Instead, use the provided `.env.example` template to manually map your configurations.

1. **Copy the environment template:**
   ```bash
   cd data_pipeline
   cp ../.env.example .env
   ```

2. **Configure your `.env` file:**
   Open the newly created `.env` file in your text editor. Do **not** use `echo` commands to append variables; map these configurations manually.

   **GCP & Storage Configuration**
   ```env
   # Your Google Cloud Project ID and GCS bucket name
   GCP_PROJECT_ID=your-gcp-project-id
   GCS_BUCKET_NAME=savvio-data-bucket
   
   # Absolute path to your GCP service account JSON key (used for local development)
   # Ex: /absolute/path/to/SavVio/data_pipeline/config/savvio-gcp-key.json
   GCP_CREDENTIALS_PATH=/path/to/your/service-account-key.json
   ```

   **Airflow & Pipeline Credentials**
   ```env
   # Get your current user ID by running `id -u` in your terminal
   AIRFLOW_UID=501
   
   # Directory to mount your pipeline to Airflow 
   AIRFLOW_PROJ_DIR=.

   # Credentials for the Airflow UI user (Used in docker-compose.yml)
   _AIRFLOW_WWW_USER_USERNAME=airflow
   _AIRFLOW_WWW_USER_PASSWORD=airflow

   # User credentials for the API/Backend (Authentication tokens sent to Airflow)
   AIRFLOW_USERNAME=airflow
   AIRFLOW_PASSWORD=airflow
   ```

   **Database Connection (PostgreSQL)**
   ```env
   # Application database credentials
   DB_USER=exampleuser
   DB_PASSWORD=examplepassword
   
   # Note: host.docker.internal reaches your Mac from inside Docker
   # In Prod, swap DB_HOST to your Cloud SQL IP/hostname
   DB_HOST=host.docker.internal
   DB_PORT=5432
   DB_NAME=examplename
   ```

   **External APIs (If API is source of data)**
   ```env
   # Authentication key for the SavVio API (if needed)
   API_KEY=your-api-key-here
   ```

   **Notifications**
   ```env
   # Slack Incoming Webhook URL for pipeline alerts (Optional) 
   # Setup: https://api.slack.com/apps → Incoming Webhooks
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

   # Your Gmail address and matching App Password for sending email notifications
   # Setup: https://support.google.com/accounts/answer/185833
   SMTP_USER=example@gmail.com
   SMTP_PASSWORD=16-digit-app-password
   ```
   
> **Production Note:** The local `.env` file approach is strictly used for local development and testing iterations. In a production environment, credential dependencies will be securely stored and accessed via the **GCP Secret Manager** service instead.

### 3. Start Airflow

The **entire data pipeline** will run via Docker Compose, with the single exception of automated tests.

1. **Initialize the database** (this will take a couple of minutes):
 
```bash
cd data_pipeline
docker compose up airflow-init
```

2. **Run the Pipeline Cluster**

```bash
cd data_pipeline
docker compose up -d
```

Wait until the terminal outputs that the services are "healthy" or "running" (e.g., you see a 200 health check response for the webserver).

### 4. Access the Airflow Web UI & Run the DAG

1. Open your browser and go to: **http://localhost:8080**
2. **Username / Password:** Use the credentials configured in your `.env` file (e.g., `AIRFLOW_USERNAME` and `AIRFLOW_PASSWORD`). The default for both is `airflow`.
3. In the Airflow UI, find your target DAG (e.g., `savvio_data_pipeline`) in the DAGs list.
4. Toggle the DAG "ON" (switch on the left).
5. Click the "Play" button (▶) to trigger a manual run.
6. Click on the DAG name to comprehensively track task progress.

---

## Stopping and Restarting

### Stop Airflow (keep data)

```bash
cd data_pipeline
docker compose down
```

### Stop Airflow (delete all data)

```bash
cd data_pipeline
docker compose down -v
```

### Restart Airflow

```bash
cd data_pipeline
docker compose up -d
```

---

## Testing

Automated tests are the only component of this project **not** orchestrated via Docker Compose.

To run the full `pytest` test suite, create a virtual environment locally:

1. Create a virtual environment and verify dependencies (including the `savviocore` local package):
```bash
python3 -m venv savvio_tests
source savvio_tests/bin/activate
pip install -e savviocore
pip install -r data_pipeline/tests/test_requirements.txt
```

2. To run all the tests:
```bash
pytest data_pipeline/tests/ -v
```

3. To run specific tests:
```bash
pytest data_pipeline/tests/bias/test_run_bias.py -v
```
