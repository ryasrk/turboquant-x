#!/usr/bin/env bash
# ── TurboQuant-X: Start n8n with workspace portal configuration ─────
set -euo pipefail

# Base URL for TurboQuant-X (default: local dev)
TQ_BASE_URL="${TQ_BASE_URL:-http://localhost:8000}"

# n8n configuration
export N8N_PATH="/workspace/n8n/"
export N8N_EDITOR_BASE_URL="${TQ_BASE_URL}/workspace/n8n/"
export N8N_HOST="${N8N_HOST:-127.0.0.1}"
export N8N_PORT="${N8N_PORT:-5678}"
export N8N_PROTOCOL="${N8N_PROTOCOL:-http}"
export WEBHOOK_URL="${TQ_BASE_URL}/workspace/n8n/"

# Trust the reverse proxy (TurboQuant-X sits in front)
export N8N_PROXY_HOPS=1

# Data persistence
export N8N_USER_FOLDER="${N8N_USER_FOLDER:-${HOME}/.n8n}"

# Security: disable public API by default; enable with N8N_PUBLIC_API=true
export N8N_PUBLIC_API_DISABLED="${N8N_PUBLIC_API_DISABLED:-true}"

# Disable n8n telemetry in dev
export N8N_DIAGNOSTICS_ENABLED="${N8N_DIAGNOSTICS_ENABLED:-false}"

# Allow iframe embedding from TurboQuant-X origin
export N8N_EDITOR_FRAME_ANCESTORS="${TQ_BASE_URL}"

# Optional: Basic auth for n8n (set via env)
# export N8N_BASIC_AUTH_ACTIVE=true
# export N8N_BASIC_AUTH_USER=admin
# export N8N_BASIC_AUTH_PASSWORD=changeme

echo "◈ Starting n8n..."
echo "  Host:       ${N8N_HOST}:${N8N_PORT}"
echo "  Editor URL: ${N8N_EDITOR_BASE_URL}"
echo "  Webhook:    ${WEBHOOK_URL}"
echo "  Data dir:   ${N8N_USER_FOLDER}"

# Detect n8n binary
if command -v n8n &>/dev/null; then
  exec n8n start
elif command -v npx &>/dev/null; then
  exec npx n8n start
else
  echo "ERROR: n8n not found. Install with: npm install -g n8n" >&2
  exit 1
fi
