"""Shared utilities for the entire data pipeline."""
import logging

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the data pipeline."""
    # Using force=True if running python 3.8+ to override existing loggers if needed
    try:
        logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", force=True)
    except ValueError:
        logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
