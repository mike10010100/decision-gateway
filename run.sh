#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

HOST="${GATEWAY_HOST:-0.0.0.0}"
PORT="${GATEWAY_PORT:-8000}"

echo "Starting Decision Gateway on http://${HOST}:${PORT}..."
exec python3 -m uvicorn app.main:app --host "$HOST" --port "$PORT" --workers 1
