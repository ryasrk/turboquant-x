#!/usr/bin/env bash
# ── TurboQuant-X Ngrok Tunnel Runner ─────────────────────────────────
# Starts the FastAPI server and exposes it via ngrok.
#
# Usage:
#   ./ngrok_run.sh              # Start server + ngrok tunnel
#   ./ngrok_run.sh --server     # Start server only (no ngrok)
#   ./ngrok_run.sh --ngrok      # Start ngrok only (server already running)
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Load .env if present ─────────────────────────────────────────────
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# ── Configuration ────────────────────────────────────────────────────
SERVER_HOST="${TURBOQUANT_HOST:-0.0.0.0}"
SERVER_PORT="${TURBOQUANT_PORT:-8000}"
NGROK_AUTH="${NGROK_AUTHTOKEN:-}"
NGROK_DOMAIN="${NGROK_DOMAIN:-}"
VENV_PYTHON="env/bin/python3"
RUN_SCRIPT="./run.sh"

# ── Theme ────────────────────────────────────────────────────────────
DIM='\033[2m'
BOLD='\033[1m'
CYAN='\033[38;5;87m'
GREEN='\033[38;5;120m'
YELLOW='\033[38;5;228m'
RED='\033[38;5;203m'
MAGENTA='\033[38;5;177m'
WHITE='\033[38;5;255m'
GRAY='\033[38;5;245m'
NC='\033[0m'

banner() {
  echo ""
  echo -e "${CYAN}${BOLD}    ⬡ ─────────────────────────────────────────── ⬡${NC}"
  echo -e "${CYAN}${BOLD}    │                                               │${NC}"
  echo -e "${CYAN}${BOLD}    │${NC}     ${WHITE}${BOLD}◈  T U R B O Q U A N T - X${NC}              ${CYAN}${BOLD}│${NC}"
  echo -e "${CYAN}${BOLD}    │${NC}        ${GRAY}neural interface · tunnel${NC}             ${CYAN}${BOLD}│${NC}"
  echo -e "${CYAN}${BOLD}    │                                               │${NC}"
  echo -e "${CYAN}${BOLD}    ⬡ ─────────────────────────────────────────── ⬡${NC}"
  echo ""
}

log()  { echo -e "  ${CYAN}◈${NC} ${WHITE}$*${NC}"; }
step() { echo -e "  ${MAGENTA}▸${NC} ${GRAY}$*${NC}"; }
ok()   { echo -e "  ${GREEN}✓${NC} ${WHITE}$*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC} ${YELLOW}$*${NC}"; }
err()  { echo -e "  ${RED}✗${NC} ${RED}$*${NC}" >&2; }

divider() {
  echo -e "  ${DIM}${CYAN}──────────────────────────────────────────────${NC}"
}

# ── Cleanup on exit ──────────────────────────────────────────────────
cleanup() {
  echo ""
  divider
  log "Shutting down neural link..."
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    step "Server terminated"
  fi
  if [[ -n "${NGROK_PID:-}" ]]; then
    kill "$NGROK_PID" 2>/dev/null || true
    wait "$NGROK_PID" 2>/dev/null || true
    step "Tunnel closed"
  fi
  ok "Session ended"
  echo ""
}
trap cleanup EXIT INT TERM

# ── Preflight checks ────────────────────────────────────────────────
check_deps() {
  step "Running preflight checks..."

  if [[ ! -x "$VENV_PYTHON" ]]; then
    err "Virtual environment not found at $VENV_PYTHON"
    err "Run: python3 -m venv env && env/bin/pip install -e ."
    exit 1
  fi
  ok "Python venv"

  if [[ ! -x "$RUN_SCRIPT" ]]; then
    err "run.sh not found or not executable at $RUN_SCRIPT"
    err "Run: chmod +x run.sh"
    exit 1
  fi
  ok "run.sh launcher"

  if ! command -v ngrok &>/dev/null; then
    err "ngrok not installed"
    step "Install: curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok-v3-stable-linux-amd64.tgz | tar xz -C /usr/local/bin"
    exit 1
  fi
  ok "ngrok binary"

  if [[ -z "$NGROK_AUTH" ]]; then
    warn "NGROK_AUTHTOKEN not set — configure in .env"
    warn "Get token: https://dashboard.ngrok.com/get-started/your-authtoken"
    exit 1
  fi
  ok "Auth token loaded"
}

# ── Start server ─────────────────────────────────────────────────────
start_server() {
  divider
  log "Initializing inference engine..."
  step "Binding ${SERVER_HOST}:${SERVER_PORT}"

  # Use the combined launcher so n8n and TurboQuant-X start together.
  TQ_HOST="$SERVER_HOST" TQ_PORT="$SERVER_PORT" "$RUN_SCRIPT" &
  SERVER_PID=$!

  # Wait for server to be ready
  step "Waiting for health check..."
  for i in $(seq 1 30); do
    if curl -sf "http://localhost:${SERVER_PORT}/health" >/dev/null 2>&1; then
      ok "Engine online (PID: ${SERVER_PID})"
      return 0
    fi
    sleep 1
  done

  err "Server failed to start within 30s"
  exit 1
}

# ── Start ngrok ──────────────────────────────────────────────────────
start_ngrok() {
  divider
  log "Establishing tunnel..."

  # Configure auth
  ngrok config add-authtoken "$NGROK_AUTH" 2>/dev/null || true

  local ngrok_args="http ${SERVER_PORT}"

  if [[ -n "$NGROK_DOMAIN" ]]; then
    ngrok_args="http --url=${NGROK_DOMAIN} ${SERVER_PORT}"
    step "Domain: ${NGROK_DOMAIN}"
  else
    step "Mode: random URL"
  fi

  ngrok $ngrok_args &
  NGROK_PID=$!
  sleep 3

  # Fetch the public URL from ngrok API
  local public_url
  public_url=$(curl -sf http://localhost:4040/api/tunnels 2>/dev/null \
    | grep -o '"public_url":"[^"]*"' \
    | head -1 \
    | cut -d'"' -f4) || true

  if [[ -n "$public_url" ]]; then
    ok "Tunnel active"
    echo ""
    echo -e "  ${CYAN}${BOLD}⬡ ─────────────────────────────────────────── ⬡${NC}"
    echo -e "  ${CYAN}${BOLD}│${NC}                                               ${CYAN}${BOLD}│${NC}"
    echo -e "  ${CYAN}${BOLD}│${NC}  ${GREEN}${BOLD}SYS:ONLINE${NC}  ${WHITE}Neural link established${NC}       ${CYAN}${BOLD}│${NC}"
    echo -e "  ${CYAN}${BOLD}│${NC}                                               ${CYAN}${BOLD}│${NC}"
    echo -e "  ${CYAN}${BOLD}│${NC}  ${GRAY}LOCAL${NC}   ${WHITE}http://localhost:${SERVER_PORT}${NC}"
    echo -e "  ${CYAN}${BOLD}│${NC}  ${GRAY}PUBLIC${NC}  ${GREEN}${BOLD}${public_url}${NC}"
    echo -e "  ${CYAN}${BOLD}│${NC}  ${GRAY}CHAT${NC}    ${GREEN}${public_url}/${NC}"
    echo -e "  ${CYAN}${BOLD}│${NC}  ${GRAY}API${NC}     ${GREEN}${public_url}/v1/chat/completions${NC}"
    echo -e "  ${CYAN}${BOLD}│${NC}  ${GRAY}DOCS${NC}    ${GREEN}${public_url}/v1/documents/list${NC}"
    echo -e "  ${CYAN}${BOLD}│${NC}                                               ${CYAN}${BOLD}│${NC}"
    echo -e "  ${CYAN}${BOLD}⬡ ─────────────────────────────────────────── ⬡${NC}"
    echo ""
  else
    warn "Could not detect public URL — check http://localhost:4040"
  fi
}

# ── Main ─────────────────────────────────────────────────────────────
main() {
  local mode="${1:-all}"

  banner

  case "$mode" in
    --server)
      check_deps
      start_server
      log "Press ${BOLD}Ctrl+C${NC} to terminate"
      wait "$SERVER_PID"
      ;;
    --ngrok)
      check_deps
      start_ngrok
      log "Press ${BOLD}Ctrl+C${NC} to terminate"
      wait "$NGROK_PID"
      ;;
    *)
      check_deps
      start_server
      start_ngrok
      log "Press ${BOLD}Ctrl+C${NC} to terminate"
      wait
      ;;
  esac
}

main "$@"
