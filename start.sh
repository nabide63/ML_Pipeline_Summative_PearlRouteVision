#!/usr/bin/env bash
# Starts the API and the Streamlit UI in one container. If either one stops, the container stops.
set -euo pipefail

API_PORT="${API_PORT:-8000}"
UI_PORT="${PORT:-8501}"   # hosts such as Render set PORT for the public (UI) port

# Create the tables and register the bundled models. Retries while a fresh Postgres is starting up.
if [ -n "${DATABASE_URL:-}" ]; then
    for attempt in $(seq 1 10); do
        if python scripts/init_db.py; then
            break
        fi
        if [ "$attempt" -eq 10 ]; then
            echo "database not reachable; starting anyway (upload and retraining will fail)" >&2
        else
            echo "database not ready (attempt $attempt/10), retrying in 3 s" >&2
            sleep 3
        fi
    done
else
    echo "DATABASE_URL is not set: prediction works, upload and retraining need a database" >&2
fi

uvicorn pearlroute.api.main:app --host 0.0.0.0 --port "$API_PORT" &
API_PID=$!

streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port "$UI_PORT" \
    --server.headless true --browser.gatherUsageStats false &
UI_PID=$!

trap 'kill "$API_PID" "$UI_PID" 2>/dev/null' TERM INT
status=0
wait -n "$API_PID" "$UI_PID" || status=$?
kill "$API_PID" "$UI_PID" 2>/dev/null || true
exit "$status"
