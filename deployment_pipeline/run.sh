#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# SavVio — Start the backend API and frontend dev server.
#
# Usage:
#   ./deployment_pipeline/run.sh            # start both (default)
#   ./deployment_pipeline/run.sh api        # backend only
#   ./deployment_pipeline/run.sh frontend   # frontend only
# -------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# -- Helpers --------------------------------------------------

red()   { printf '\033[0;31m%s\033[0m\n' "$*"; }
green() { printf '\033[0;32m%s\033[0m\n' "$*"; }
info()  { printf '\033[0;36m[info]\033[0m %s\n' "$*"; }

cleanup() {
    info "Shutting down..."
    # Kill background jobs (API server, frontend dev server)
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Backend API ──────────────────────────────────────────────

start_api() {
    info "Starting backend API on http://localhost:3500"

    # Activate model_pipeline venv if it exists and we're not already in one
    local venv="$ROOT/model_pipeline/.venv/bin/activate"
    if [[ -z "${VIRTUAL_ENV:-}" && -f "$venv" ]]; then
        info "Activating model_pipeline virtualenv"
        # shellcheck disable=SC1090
        source "$venv"
    fi

    export PYTHONPATH="${ROOT}/model_pipeline/src:${ROOT}/savviocore/src:${ROOT}"
    uvicorn deployment_pipeline.api.main:app \
        --host 0.0.0.0 \
        --port 3500 \
        --reload \
        "$@"
}

# ── Frontend Dev Server ──────────────────────────────────────

start_frontend() {
    info "Starting frontend dev server on http://localhost:3000"

    cd "$ROOT/deployment_pipeline/frontend"

    if [[ ! -d node_modules ]]; then
        info "Installing frontend dependencies..."
        npm install
    fi

    npx vite --host 0.0.0.0 --port 3000
}

# ── Entrypoint ───────────────────────────────────────────────

MODE="${1:-both}"

case "$MODE" in
    api|backend)
        shift || true
        start_api "$@"
        ;;
    frontend|ui)
        start_frontend
        ;;
    both|"")
        start_api &
        API_PID=$!
        # Give the API a moment to bind the port
        sleep 2
        start_frontend &
        FE_PID=$!
        info "Backend PID=$API_PID  |  Frontend PID=$FE_PID"
        wait
        ;;
    *)
        red "Unknown mode: $MODE"
        echo "Usage: $0 [api|frontend|both]"
        exit 1
        ;;
esac
