#!/bin/sh
# Start the demo at http://localhost:8000
cd "$(dirname "$0")/probe" && exec ../.venv/bin/uvicorn server:app --port "${PORT:-8000}"
