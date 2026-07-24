#!/bin/bash
set -Eeuo pipefail

DATA_PATH="${DATA_PATH:-/data}"
PORT="${PORT:-8080}"

echo "Starting Lance Viewer on port ${PORT}..."
if [ -d "$DATA_PATH" ] && [ -r "$DATA_PATH" ]; then
    echo "Data path: $DATA_PATH"
else
    echo "WARNING: Mounted datasets are unavailable; remote dataset URIs can still be opened"
fi

exec python -m uvicorn app:app --host 0.0.0.0 --port "${PORT}"