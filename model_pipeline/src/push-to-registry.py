"""
Push Model to GCP Artifact Registry.

Standalone script called by CI/CD (GitHub Actions Job 4) after all
gates pass. Not called during local development.

Pulls the latest registered model from MLflow, serializes it, and
uploads it to GCP Artifact Registry with full lineage metadata.

Usage (CI/CD):
    python scripts/push_to_registry.py

Requires:
    - GOOGLE_APPLICATION_CREDENTIALS env var set (service account key)
    - gcloud CLI authenticated
    - MLflow tracking server accessible
"""

import os
import logging
import shutil
import subprocess
import tempfile

import mlflow
from mlflow.tracking import MlflowClient

from config import Config

logger = logging.getLogger(__name__)


def get_champion_model_version(mlflow_client, registered_model_name):
    """Return the model version object for the 'champion' alias.
    Raises:
        RuntimeError: if no champion alias exists for the registered model.
    """
    try:
        return mlflow_client.get_model_version_by_alias(registered_model_name, "champion")
    except Exception:
        logger.error("No champion model found for '%s' in MLflow registry.", registered_model_name)
        raise RuntimeError(f"No best/champion model available for '{registered_model_name}' in MLflow registry")


def download_registered_model_locally(model_name, destination_dir):
    """Download model artifact from MLflow registry using champion alias."""
    champion_model_uri = f"models:/{model_name}@champion"
    return mlflow.artifacts.download_artifacts(
        artifact_uri=champion_model_uri,
        dst_path=destination_dir,
    )


def push_model_to_gcp(model_archive_path, registered_model_name, model_version):
    """Upload champion model artifact to GCP Artifact Registry."""
    repository_name = Config.ARTIFACT_REGISTRY_REPO
    gcp_location = Config.GCP_REGION
    gcp_project_id = Config.GCP_PROJECT_ID
    package_name = registered_model_name.lower().replace(" ", "-")

    # Ensure repository exists (idempotent).
    subprocess.run([
        "gcloud", "artifacts", "repositories", "describe",
        repository_name, f"--location={gcp_location}", f"--project={gcp_project_id}",
    ], capture_output=True)

    # Upload model artifact.
    logger.info("Pushing model to GCP Artifact Registry...")
    upload_result = subprocess.run([
        "gcloud", "artifacts", "generic", "upload",
        f"--repository={repository_name}",
        f"--location={gcp_location}",
        f"--project={gcp_project_id}",
        f"--package={package_name}",
        f"--version={model_version}",
        f"--source={model_archive_path}",
    ], capture_output=True, text=True)

    if upload_result.returncode != 0:
        logger.error("GCP push failed: %s", upload_result.stderr)
        raise RuntimeError(f"gcloud upload failed: {upload_result.stderr}")

    logger.info("Successfully pushed %s v%s to GCP Artifact Registry.", package_name, model_version)


def main():
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow_client = MlflowClient(tracking_uri=Config.MLFLOW_TRACKING_URI)

    # 1. Resolve champion model from registry.
    champion_model_version = get_champion_model_version(mlflow_client, Config.REGISTERED_MODEL_NAME)

    with tempfile.TemporaryDirectory() as temp_directory:
        # 2. Download champion model artifact from MLflow.
        downloaded_model_path = download_registered_model_locally(
            Config.REGISTERED_MODEL_NAME,
            temp_directory,
        )

        # 3. Archive the downloaded model directory for reliable generic upload.
        model_archive_path = shutil.make_archive(
            base_name=os.path.join(temp_directory, "champion_model"),
            format="gztar",
            root_dir=downloaded_model_path,
        )

        # 4. Push to GCP Artifact Registry.
        push_model_to_gcp(
            model_archive_path,
            champion_model_version.name,
            champion_model_version.version,
        )

    logger.info("Registry push complete.")


if __name__ == "__main__":
    main()
