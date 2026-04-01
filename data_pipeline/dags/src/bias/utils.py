"""Shared utilities for bias detection."""
import logging
import os

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


from src.utils import setup_logging


def get_processed_path(filename: str, base_dir: str = None) -> str:
    """Return path to a processed data file."""
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    return os.path.join(base_dir, "data/processed", filename)


def get_features_path(filename: str, base_dir: str = None) -> str:
    """Return path to a feature-engineered data file."""
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    return os.path.join(base_dir, "data/features", filename)
