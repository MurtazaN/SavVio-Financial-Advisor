# latest Airflow image
FROM apache/airflow:3.1.7

COPY data_pipeline/requirements.txt .

# Install CPU-only PyTorch BEFORE requirements so sentence-transformers reuses it
# instead of pulling the full CUDA build (~2GB) which is useless on a CPU-only VM.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=airflow:0 savviocore /opt/airflow/savviocore
# TODO: Remove '-e' flag for production deployments so savviocore is baked into site-packages
RUN pip install --no-cache-dir -e /opt/airflow/savviocore
