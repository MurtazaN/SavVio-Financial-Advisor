"""
Collect current production data for drift detection.

Queries the production database (financial_profiles + products tables)
and returns a DataFrame (or optionally saves to CSV).

Usage:
    # As a library (used by drift_detector.py):
    from collect_production_data import collect_df
    df = collect_df()

    # As a script (saves CSV for inspection):
    python deployment_pipeline/monitoring/drift/collect_production_data.py
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MONITORING  = Path(__file__).parent.parent     # deployment_pipeline/monitoring/
_CONFIG_PATH = Path(__file__).parent / "alert_config.yaml"

# Add savviocore to path so get_engine is importable
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "savviocore" / "src"))

# ---------------------------------------------------------------------------
# Load monitored feature list from shared config
# ---------------------------------------------------------------------------
with open(_CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

FINANCIAL_COLS = _cfg["monitored_features"]["financial"]
PRODUCT_COLS   = _cfg["monitored_features"]["product"]
MONITOR_COLS   = FINANCIAL_COLS + PRODUCT_COLS

# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------
def collect_df(sample_size: int = 10_000) -> pd.DataFrame:
    """Query DB and return current production data as a DataFrame."""
    try:
        from savviocore.database.db_connection import get_engine
    except ImportError as e:
        print(f"ERROR: Cannot import database connection: {e}")
        print("Make sure savviocore is installed.")
        sys.exit(1)

    engine = get_engine()

    fin_cols = ", ".join(FINANCIAL_COLS)
    financial_df = pd.read_sql(f"SELECT {fin_cols} FROM financial_profiles", engine)

    prod_cols = ", ".join(PRODUCT_COLS)
    products_df = pd.read_sql(f"SELECT {prod_cols} FROM products", engine)

    # Cross-sample to mirror training data distribution (user × product pairs).
    n = min(sample_size, len(financial_df), len(products_df))
    fin_sample  = financial_df.sample(n=n, random_state=None).reset_index(drop=True)
    prod_sample = products_df.sample(n=n, random_state=None).reset_index(drop=True)

    return pd.concat([fin_sample, prod_sample], axis=1)[MONITOR_COLS].dropna()


if __name__ == "__main__":
    # Save to CSV for manual inspection only — not required for drift detection.
    output = _MONITORING / "data" / "current_production_data.csv"
    df = collect_df()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved: {output}  ({len(df):,} rows, {len(MONITOR_COLS)} columns)")
