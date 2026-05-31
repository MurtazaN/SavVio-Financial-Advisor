"""
Generate baseline_data.csv for drift detection and upload to GCS.

Reads the training dataset, extracts monitored feature columns, and uploads
the baseline to GCS. Run once after initial setup or whenever the model is
retrained on new data:

    python deployment_pipeline/monitoring/drift/generate_baseline.py

GCS destination: gs://savvio-dev-mlflow-artifacts/monitoring/baseline_data.csv
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
_REPO_ROOT   = Path(__file__).parent.parent.parent.parent   # SavVio/
_CONFIG_PATH = Path(__file__).parent / "alert_config.yaml"

_TRAINING_DATA = _REPO_ROOT / "model_pipeline" / "data" / "training_scenarios.csv"
_GCS_BASELINE  = "gs://savvio-dev-mlflow-artifacts/monitoring/baseline_data.csv"

with open(_CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

MONITOR_COLS = (
    _cfg["monitored_features"]["financial"]
    + _cfg["monitored_features"]["product"]
)

# ---------------------------------------------------------------------------
# Generate and upload
# ---------------------------------------------------------------------------
def generate(sample_size: int = 10_000) -> None:
    if not _TRAINING_DATA.exists():
        print(f"ERROR: Training data not found at {_TRAINING_DATA}")
        print("Check that model_pipeline/data/training_scenarios.csv exists.")
        sys.exit(1)

    df = pd.read_csv(_TRAINING_DATA)

    missing = [c for c in MONITOR_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: Monitored columns missing from training data (will be skipped): {missing}")

    cols     = [c for c in MONITOR_COLS if c in df.columns]
    baseline = df[cols].dropna()
    if len(baseline) > sample_size:
        baseline = baseline.sample(n=sample_size, random_state=42)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        baseline.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    print(f"Uploading baseline ({len(baseline):,} rows, {len(cols)} cols) → {_GCS_BASELINE}")
    result = subprocess.run(
        ["gcloud", "storage", "cp", tmp_path, _GCS_BASELINE],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: GCS upload failed:\n{result.stderr}")
        sys.exit(1)

    Path(tmp_path).unlink(missing_ok=True)
    print(f"Baseline uploaded successfully → {_GCS_BASELINE}")
    print(f"Columns: {cols}")


if __name__ == "__main__":
    generate()
