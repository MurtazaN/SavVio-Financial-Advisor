"""Shared utilities for preprocessing."""
import logging
import os

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


from src.utils import setup_logging


def get_raw_path(filename: str, base_dir: str = "data/raw") -> str:
    """Return path to a raw data file (relative to data_pipeline)."""
    return os.path.join(base_dir, filename)


def get_processed_path(filename: str, base_dir: str = "data/processed") -> str:
    """Return path to a processed data file (relative to data_pipeline)."""
    return os.path.join(base_dir, filename)


def ensure_output_dir(file_path: str) -> None:
    """Ensure the directory for file_path exists."""
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
