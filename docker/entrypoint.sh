#!/bin/bash
set -Eeuo pipefail

PORT="${PORT:-8080}"

echo "Starting Lance Viewer on port ${PORT}..."

exec python -m uvicorn app:app --host 0.0.0.0 --port "${PORT}"