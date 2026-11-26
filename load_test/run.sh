#!/usr/bin/env bash
# Runs the load test at 10, 50 and 100 simulated users and saves CSV stats + an HTML report per level.
#   bash load_test/run.sh                                   # against the local container
#   bash load_test/run.sh https://<user>-<space>.hf.space   # against the hosted API
set -euo pipefail

HOST="${1:-http://localhost:8000}"
DURATION="${DURATION:-90s}"
OUT="reports/locust"
LOCUST="${LOCUST:-locust}"
[ -x .venv/Scripts/locust ] && LOCUST=.venv/Scripts/locust
[ -x .venv/bin/locust ] && LOCUST=.venv/bin/locust
mkdir -p "$OUT"

for users in 10 50 100; do
    echo "== $users users for $DURATION against $HOST"
    "$LOCUST" -f load_test/locustfile.py --host "$HOST" --headless \
        -u "$users" -r 10 -t "$DURATION" --only-summary \
        --csv "$OUT/u$users" --html "$OUT/u$users.html" \
        || echo "(some requests failed at $users users; see $OUT/u${users}_failures.csv)"
done
echo "results in $OUT/"
