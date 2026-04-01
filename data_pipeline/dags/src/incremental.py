"""
Incremental merge utilities for the SavVio data pipeline.

Provides reusable functions for merging new data with existing data
at every stage (preprocessing, features). When a record with the same
key exists in both old and new data, the new version wins. New records
are appended, and old records not present in the new data are kept.
"""

import hashlib
import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File checksum
# ---------------------------------------------------------------------------

def file_checksum(path: str) -> str:
    """
    Compute MD5 checksum of a file.

    Uses MD5 for consistency with GCS blob checksums.

    Args:
        path: Path to the file.

    Returns:
        Hex-encoded MD5 digest string.
    """
    return hashlib.md5(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------------------
# CSV merge
# ---------------------------------------------------------------------------

def merge_csv(
    new_path: str,
    existing_path: str,
    key_cols: List[str],
) -> Dict[str, int]:
    """
    Merge a newly-produced CSV with an existing CSV file out-of-core using DuckDB.
    
    Logic:
        - Records whose key exists in both → replaced with the new version.
        - Records only in the new file → appended.
        - Records only in the existing file → kept unchanged.
    """
    logger.info("Starting out-of-core CSV merge using DuckDB...")

    if not os.path.exists(existing_path) or os.path.getsize(existing_path) == 0:
        # First run — nothing to merge with.
        shutil.copy(new_path, existing_path)
        with duckdb.connect() as con:
            total_res = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{new_path}')").fetchone()
            total = total_res[0] if total_res else 0
        stats = {"updated": 0, "appended": total, "unchanged": 0, "total": total}
        logger.info("First run — wrote %d records: %s", total, existing_path)
        return stats

    partition_clause = ", ".join(key_cols)
    temp_out = existing_path + ".duckdb.tmp"

    # Set DuckDB configuration
    # Memory limit - Increase when docker has access to more memory 
    # Threads - Increase when docker has access to more CPU
    # preserve_insertion_order=false - order of records is not preserved - needs less memory
    # temp_directory='/tmp' - duckdb instead of using memory, saves partial work to /tmp and reads back from /tmp
    with duckdb.connect() as con:
        con.execute("PRAGMA temp_directory='/tmp';")
        con.execute("PRAGMA memory_limit='2GB';")
        con.execute("PRAGMA preserve_insertion_order=false;")
        con.execute("PRAGMA threads=2;")

        new_count_res = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{new_path}')").fetchone()
        new_count = new_count_res[0] if new_count_res else 0
        
        existing_count_res = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{existing_path}')").fetchone()
        existing_count = existing_count_res[0] if existing_count_res else 0
        
        logger.info("New CSV records: %d | Existing CSV records: %d", new_count, existing_count)

        query = f"""
        COPY (
            WITH new_data AS (
                SELECT *, 1 AS __source_priority FROM read_csv_auto('{new_path}')
            ),
            existing_data AS (
                SELECT *, 2 AS __source_priority FROM read_csv_auto('{existing_path}')
            ),
            combined AS (
                SELECT * FROM new_data
                UNION ALL BY NAME
                SELECT * FROM existing_data
            ),
            deduped AS (
                SELECT * EXCLUDE (__source_priority, __rn)
                FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY {partition_clause} ORDER BY __source_priority ASC) AS __rn
                    FROM combined
                )
                WHERE __rn = 1
            )
            SELECT * FROM deduped
        ) TO '{temp_out}' (FORMAT CSV, HEADER);
        """
        con.execute(query)
        
        total_count_res = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{temp_out}')").fetchone()
        total_count = total_count_res[0] if total_count_res else 0

    appended = total_count - existing_count
    updated = new_count - appended
    unchanged = existing_count - updated

    stats = {
        "updated": updated,
        "appended": appended,
        "unchanged": unchanged,
        "total": total_count,
    }

    os.replace(temp_out, existing_path)
    logger.info("Merged CSV → %s | %s", existing_path, stats)
    return stats


# ---------------------------------------------------------------------------
# JSONL merge
# ---------------------------------------------------------------------------

def merge_jsonl(
    new_path: str,
    existing_path: str,
    key_cols: List[str],
) -> Dict[str, int]:
    """
    Merge a newly-produced JSONL with an existing JSONL file out-of-core using DuckDB.
    
    Same logic as merge_csv but for JSONL files (one JSON object per line).
    """
    logger.info("Starting out-of-core JSONL merge using DuckDB...")

    if not os.path.exists(existing_path) or os.path.getsize(existing_path) == 0:
        # First run — nothing to merge with.
        shutil.copy(new_path, existing_path)
        with duckdb.connect() as con:
            total_res = con.execute(f"SELECT COUNT(*) FROM read_json_auto('{new_path}')").fetchone()
            total = total_res[0] if total_res else 0
        stats = {"updated": 0, "appended": total, "unchanged": 0, "total": total}
        logger.info("First run — wrote %d records: %s", total, existing_path)
        return stats

    partition_clause = ", ".join(key_cols)
    temp_out = existing_path + ".duckdb.tmp"

    with duckdb.connect() as con:
        con.execute("PRAGMA temp_directory='/tmp';")
        con.execute("PRAGMA memory_limit='2GB';")
        con.execute("PRAGMA preserve_insertion_order=false;")
        con.execute("PRAGMA threads=2;")

        new_count_res = con.execute(f"SELECT COUNT(*) FROM read_json_auto('{new_path}')").fetchone()
        new_count = new_count_res[0] if new_count_res else 0
        
        existing_count_res = con.execute(f"SELECT COUNT(*) FROM read_json_auto('{existing_path}')").fetchone()
        existing_count = existing_count_res[0] if existing_count_res else 0
        
        logger.info("New JSONL records: %d | Existing JSONL records: %d", new_count, existing_count)

        query = f"""
        COPY (
            WITH new_data AS (
                SELECT *, 1 AS __source_priority FROM read_json_auto('{new_path}', maximum_object_size=33554432)
            ),
            existing_data AS (
                SELECT *, 2 AS __source_priority FROM read_json_auto('{existing_path}', maximum_object_size=33554432)
            ),
            combined AS (
                SELECT * FROM new_data
                UNION ALL BY NAME
                SELECT * FROM existing_data
            ),
            deduped AS (
                SELECT * EXCLUDE (__source_priority, __rn)
                FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY {partition_clause} ORDER BY __source_priority ASC) AS __rn
                    FROM combined
                )
                WHERE __rn = 1
            )
            SELECT * FROM deduped
        ) TO '{temp_out}' (FORMAT JSON);
        """
        con.execute(query)
        
        total_count_res = con.execute(f"SELECT COUNT(*) FROM read_json_auto('{temp_out}')").fetchone()
        total_count = total_count_res[0] if total_count_res else 0

    appended = total_count - existing_count
    updated = new_count - appended
    unchanged = existing_count - updated

    stats = {
        "updated": updated,
        "appended": appended,
        "unchanged": unchanged,
        "total": total_count,
    }

    os.replace(temp_out, existing_path)
    logger.info("Merged JSONL → %s | %s", existing_path, stats)
    return stats
