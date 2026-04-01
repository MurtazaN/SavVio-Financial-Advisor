"""
Shared validation infrastructure for SavVio data pipeline.

Provides:
  - Severity levels (INFO / WARNING / CRITICAL)
  - Configurable thresholds for each level
  - Result aggregation and reporting
  - Alert dispatching based on severity
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

class Severity(IntEnum):
    """Pipeline action depends on severity of validation failures."""
    INFO = 0       # Log only — minor issues
    WARNING = 1    # Email/Slack alert — notable but non-blocking
    CRITICAL = 2   # Halt pipeline — data integrity at risk


# ---------------------------------------------------------------------------
# Individual validation result
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Outcome of a single expectation / validation check."""
    check_name: str
    passed: bool
    severity: Severity
    dataset: str          # "financial", "products", or "reviews"
    stage: str            # "raw", "processed", or "features"
    details: str = ""     # Human-readable context
    metric_value: Any = None  # e.g. actual null %, actual min value

    @property
    def tag(self) -> str:
        return f"[{self.severity.name}]"


# ---------------------------------------------------------------------------
# Aggregated validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Collects all check results for a pipeline stage."""
    stage: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    results: list[CheckResult] = field(default_factory=list)

    # --- helpers -----------------------------------------------------------

    def add(self, result: CheckResult) -> None:
        self.results.append(result)
        level = logging.INFO if result.passed else {
            Severity.INFO: logging.INFO,
            Severity.WARNING: logging.WARNING,
            Severity.CRITICAL: logging.ERROR,
        }[result.severity]
        status = "PASS" if result.passed else f"FAIL {result.tag}"
        logger.log(level, "%s  %s — %s  %s", status, result.check_name,
                   result.details, f"(value={result.metric_value})" if result.metric_value is not None else "")

    @property
    def passed(self) -> bool:
        return not any(r.severity == Severity.CRITICAL and not r.passed for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.severity >= Severity.WARNING and not r.passed for r in self.results)

    @property
    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed_info = sum(1 for r in self.results if not r.passed and r.severity == Severity.INFO)
        failed_warn = sum(1 for r in self.results if not r.passed and r.severity == Severity.WARNING)
        failed_crit = sum(1 for r in self.results if not r.passed and r.severity == Severity.CRITICAL)
        return {
            "stage": self.stage,
            "timestamp": self.timestamp,
            "total_checks": total,
            "passed": passed,
            "failed_info": failed_info,
            "failed_warning": failed_warn,
            "failed_critical": failed_crit,
            "pipeline_action": "HALT" if failed_crit > 0 else ("ALERT" if failed_warn > 0 else "CONTINUE"),
        }

    def save(self, log_dir: str = "logs/validation") -> str:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"{self.stage}_validation_{ts}.json")
        payload = {
            "summary": self.summary,
            "checks": [asdict(r) for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info("Validation report saved → %s", path)
        return path

    def print_summary(self) -> None:
        s = self.summary
        print(f"\n{'='*60}")
        print(f"  Validation Report: {s['stage'].upper()}")
        print(f"  {s['timestamp']}")
        print(f"{'='*60}")
        print(f"  Total checks : {s['total_checks']}")
        print(f"  Passed       : {s['passed']}")
        print(f"  Failed INFO  : {s['failed_info']}")
        print(f"  Failed WARN  : {s['failed_warning']}")
        print(f"  Failed CRIT  : {s['failed_critical']}")
        print(f"  Action       : {s['pipeline_action']}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Threshold configuration (override via env vars or config file)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    # Maximum allowed null percentage per column before triggering severity
    "null_pct_info": 0.01,        # > 1% nulls → INFO
    "null_pct_warning": 0.05,     # > 5% nulls → WARNING
    "null_pct_critical": 0.20,    # > 20% nulls → CRITICAL

    # Minimum required records
    "min_records_warning": 100,
    "min_records_critical": 10,

    # Duplicate percentage thresholds
    "dup_pct_info": 0.01,
    "dup_pct_warning": 0.05,
    "dup_pct_critical": 0.10,
}


def load_thresholds(config_path: Optional[str] = None) -> dict:
    """Load thresholds from JSON file, falling back to defaults."""
    thresholds = DEFAULT_THRESHOLDS.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            overrides = json.load(f)
        thresholds.update(overrides)
        logger.info("Loaded threshold overrides from %s", config_path)
    return thresholds
