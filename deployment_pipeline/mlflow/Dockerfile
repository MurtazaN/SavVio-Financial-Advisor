FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==2.21.3 \
    psycopg2-binary \
    google-cloud-storage

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["/entrypoint.sh"]
