FROM python:3.11-slim
WORKDIR /app

# System deps: libgomp1 for XGBoost, libpq-dev + gcc for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (cache-friendly layer)
COPY deployment_pipeline/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# savviocore source (matches APIConfig.SAVVIOCORE_SRC = /app/savviocore/src)
COPY savviocore/src/ /app/savviocore/src/

# model_pipeline/src — features/, llm/, core_models/ needed at inference time
COPY model_pipeline/src/ /app/model_pipeline/src/

# Bundled model artifacts — the pyfunc directory loaded by model_loader.py
# No runtime MLflow server dependency: model is self-contained.
COPY model_pipeline/models/model/ /app/model_pipeline/models/model/

# Deployment package (api code + __init__.py for package imports)
COPY deployment_pipeline/__init__.py /app/deployment_pipeline/__init__.py
COPY deployment_pipeline/api/ /app/deployment_pipeline/api/

ENV PYTHONPATH="/app/model_pipeline/src:/app/savviocore/src:/app"
ENV API_HOST="0.0.0.0"
ENV API_PORT="8080"
# MODEL_ARTIFACT_DIR defaults to /app/model_pipeline/models/model via config.py — no override needed.

EXPOSE 8080
CMD ["uvicorn", "deployment_pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
