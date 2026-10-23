#!/usr/bin/env bash
# Run a k6 harness N times, each against a fresh router + fresh SQLite DB,
# and write per-run + median results to JSON.
#   usage: scripts/bench_runs.sh <generate|chat> <runs> <out.json>
# Backends must already be up (`make services`). Stop any router on :8000 first.
set -euo pipefail
cd "$(dirname "$0")/.."

HARNESS=${1:?harness: generate|chat}
RUNS=${2:-3}
OUT=${3:?output json path}
case "$HARNESS" in
  generate) SCRIPT=bench/k6.js ;;
  chat) SCRIPT=bench/k6_chat.js ;;
  *) echo "unknown harness: $HARNESS" >&2; exit 2 ;;
esac

curl -sf localhost:9001/healthz >/dev/null || { echo "backends not up; run make services" >&2; exit 1; }
if curl -sf localhost:8000/v1/conversations >/dev/null 2>&1; then
  echo "something is already serving :8000; stop it first" >&2; exit 1
fi

WORK=$(mktemp -d)
ROUTER_PID=""
cleanup() { [[ -n "$ROUTER_PID" ]] && kill "$ROUTER_PID" 2>/dev/null || true; }
trap cleanup EXIT

for i in $(seq 1 "$RUNS"); do
  APP_DB_PATH="$WORK/run$i.db" uv run uvicorn app.main:app --port 8000 >"$WORK/router$i.log" 2>&1 &
  ROUTER_PID=$!
  for _ in $(seq 1 30); do
    curl -sf localhost:8000/v1/conversations >/dev/null 2>&1 && break
    sleep 1
  done
  echo "--- $HARNESS run $i/$RUNS ($SCRIPT)" >&2
  k6 run --quiet --summary-export "$WORK/summary$i.json" "$SCRIPT" >"$WORK/stdout$i.txt" 2>"$WORK/stderr$i.txt" || true
  cat "$WORK/stdout$i.txt" >&2
  kill "$ROUTER_PID"; wait "$ROUTER_PID" 2>/dev/null || true
  ROUTER_PID=""
  sleep 2
done

python3 scripts/bench_results.py collect "$HARNESS" "$SCRIPT" "$WORK" "$RUNS" "$OUT"
echo "wrote $OUT" >&2
