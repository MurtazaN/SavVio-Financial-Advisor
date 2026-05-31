#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/model_pipeline/src:$ROOT/savviocore/src:$ROOT"
exec uvicorn deployment_pipeline.api.main:app --host 0.0.0.0 --port 3500 --reload "$@"
