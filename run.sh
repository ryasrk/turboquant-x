#!/usr/bin/env bash
# ── TurboQuant-X + n8n: Combined startup script ─────────────────────
#
# Usage:
#   ./run.sh                    # Start both TurboQuant-X and n8n
#   ./run.sh --tq-only          # Start TurboQuant-X only (no n8n)
#   ./run.sh --n8n-only         # Start n8n only
#   ./run.sh --config cloud.yaml # Custom TurboQuant-X config
#
# Environment:
#   TQ_HOST          Server bind address   (default: 0.0.0.0)
#   TQ_PORT          Server port           (default: 8000)
#   TQ_CONFIG        Config file           (default: config/cloud.yaml)
#   N8N_PORT         n8n port              (default: 5678)
#   N8N_API_KEY      n8n API key for auth bridge
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ────────────────────────────────────────────────────────
TQ_HOST="${TQ_HOST:-0.0.0.0}"
TQ_PORT="${TQ_PORT:-8000}"
TQ_CONFIG="${TQ_CONFIG:-config/cloud.yaml}"
N8N_PORT="${N8N_PORT:-5678}"
TQ_BASE_URL="${TQ_BASE_URL:-http://localhost:${TQ_PORT}}"

# ── Parse arguments ─────────────────────────────────────────────────
START_TQ=true
START_N8N=true
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --tq-only)   START_N8N=false ;;
    --n8n-only)  START_TQ=false ;;
    --config)    shift; TQ_CONFIG="$1" ;;
    --config=*)  TQ_CONFIG="${arg#--config=}" ;;
    *)           EXTRA_ARGS+=("$arg") ;;
  esac
done

# ── Port cleanup (restart safety) ──────────────────────────────────
find_listen_pids() {
  local port="$1"

  if command -v lsof &>/dev/null; then
    lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -u || true
    return 0
  fi

  ss -lptn "sport = :${port}" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
    | sort -u || true
  return 0
}

free_port_if_needed() {
  local port="$1"
  local service_name="$2"
  local pids

  pids="$(find_listen_pids "$port" | tr '\n' ' ')"
  if [[ -z "${pids// }" ]]; then
    return
  fi

  echo "◈ Port ${port} busy (${service_name}), stopping existing process(es): ${pids}"

  # Graceful stop first.
  kill ${pids} 2>/dev/null || true
  sleep 1

  # Force stop only if still listening.
  pids="$(find_listen_pids "$port" | tr '\n' ' ')"
  if [[ -n "${pids// }" ]]; then
    kill -9 ${pids} 2>/dev/null || true
  fi
}

kill_matching_processes() {
  local pattern="$1"
  local label="$2"
  local pids

  pids="$(pgrep -f "$pattern" 2>/dev/null | tr '\n' ' ' || true)"
  if [[ -z "${pids// }" ]]; then
    return
  fi

  echo "◈ Existing ${label} process(es) detected, stopping: ${pids}"
  kill ${pids} 2>/dev/null || true
  sleep 1

  pids="$(pgrep -f "$pattern" 2>/dev/null | tr '\n' ' ' || true)"
  if [[ -n "${pids// }" ]]; then
    kill -9 ${pids} 2>/dev/null || true
  fi
}

# ── Detect Python ───────────────────────────────────────────────────
if [[ -x env/bin/python3 ]]; then
  PYTHON=env/bin/python3
elif command -v python3 &>/dev/null; then
  PYTHON=python3
else
  echo "ERROR: Python 3 not found." >&2
  exit 1
fi

# ── Cleanup on exit ─────────────────────────────────────────────────
PIDS=()
cleanup() {
  echo ""
  echo "◈ Shutting down..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  echo "◈ Done."
}
trap cleanup EXIT INT TERM

# ── Start n8n ───────────────────────────────────────────────────────
if [[ "$START_N8N" == true ]]; then
  kill_matching_processes "n8n start" "n8n"
  free_port_if_needed "$N8N_PORT" "n8n"
  echo "◈ Starting n8n on port ${N8N_PORT}..."

  export N8N_PATH="/workspace/n8n/"
  export N8N_EDITOR_BASE_URL="${TQ_BASE_URL}/workspace/n8n/"
  export N8N_HOST="127.0.0.1"
  export N8N_PORT="${N8N_PORT}"
  export N8N_PROTOCOL="http"
  export WEBHOOK_URL="${TQ_BASE_URL}/workspace/n8n/"
  export N8N_PROXY_HOPS=1
  export N8N_USER_FOLDER="${N8N_USER_FOLDER:-${HOME}/.n8n}"
  export N8N_PUBLIC_API_DISABLED="${N8N_PUBLIC_API_DISABLED:-false}"
  export N8N_DIAGNOSTICS_ENABLED="${N8N_DIAGNOSTICS_ENABLED:-false}"
  export N8N_EDITOR_FRAME_ANCESTORS="${TQ_BASE_URL}"

  if command -v n8n &>/dev/null; then
    n8n start &
    PIDS+=($!)
  elif command -v npx &>/dev/null; then
    # Check if we have stored auto-provisioned credentials
    CREDS_FILE="${HOME}/.turboquant/n8n_credentials.json"
    if [[ ! -f "$CREDS_FILE" ]]; then
      # First time or lost credentials — reset n8n user management
      echo "  No stored n8n credentials, resetting user management..."
      npx n8n user-management:reset 2>/dev/null || true
    fi
    npx n8n start &
    PIDS+=($!)
  else
    echo "WARNING: n8n not found. Install with: npm install -g n8n"
    echo "         Workspace features will be unavailable."
    START_N8N=false
  fi

  if [[ "$START_N8N" == true ]]; then
    # Wait briefly for n8n to start
    echo "  Waiting for n8n to be ready..."
    for i in $(seq 1 30); do
      if curl -sf "http://127.0.0.1:${N8N_PORT}/healthz" >/dev/null 2>&1; then
        echo "  n8n ready on port ${N8N_PORT}"
        break
      fi
      if [[ $i -eq 30 ]]; then
        echo "  WARNING: n8n did not respond within 30s (may still be starting)"
      fi
      sleep 1
    done

    # Auto-provision n8n owner account (no manual setup needed)
    echo "  Auto-provisioning n8n owner account..."
    export N8N_BACKEND_URL="http://127.0.0.1:${N8N_PORT}"
    N8N_SETUP_OK=$($PYTHON -c "
import asyncio
from src.server.n8n_setup import ensure_n8n_ready
result = asyncio.run(ensure_n8n_ready())
print('OK' if result else 'FAIL')
" 2>&1 | tail -1)

    if [[ "$N8N_SETUP_OK" == "OK" ]]; then
      echo "  n8n auto-setup: ready"
    else
      echo "  n8n auto-setup: first attempt failed, resetting user management..."
      rm -f "${HOME}/.turboquant/n8n_credentials.json"
      npx n8n user-management:reset 2>/dev/null || true
      # Retry provisioning after reset
      N8N_RETRY=$($PYTHON -c "
import asyncio
from src.server.n8n_setup import ensure_n8n_ready
result = asyncio.run(ensure_n8n_ready())
print('OK' if result else 'FAIL')
" 2>&1 | tail -1)
      if [[ "$N8N_RETRY" == "OK" ]]; then
        echo "  n8n auto-setup: ready (after reset)"
      else
        echo "  WARNING: n8n auto-setup failed. Set N8N_API_KEY or configure manually."
      fi
    fi
  fi
fi

# ── Start TurboQuant-X ──────────────────────────────────────────────
if [[ "$START_TQ" == true ]]; then
  kill_matching_processes "src.main.*--port ${TQ_PORT}" "TurboQuant-X"
  free_port_if_needed "$TQ_PORT" "TurboQuant-X"
  echo "◈ Starting TurboQuant-X on ${TQ_HOST}:${TQ_PORT}..."
  echo "  Config: ${TQ_CONFIG}"

  if [[ "$START_N8N" == true ]]; then
    export N8N_BACKEND_URL="http://127.0.0.1:${N8N_PORT}"
  fi

  $PYTHON -m src.main --config "$TQ_CONFIG" --host "$TQ_HOST" --port "$TQ_PORT" "${EXTRA_ARGS[@]}" &
  PIDS+=($!)

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  TurboQuant-X:  http://localhost:${TQ_PORT}"
  echo "  Workspaces:    http://localhost:${TQ_PORT}/workspaces"
  if [[ "$START_N8N" == true ]]; then
    echo "  n8n (proxied): http://localhost:${TQ_PORT}/workspace/n8n/"
  fi
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
fi

# ── Wait for all background processes ────────────────────────────────
wait "${PIDS[@]}"
