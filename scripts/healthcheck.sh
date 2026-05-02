#!/usr/bin/env bash
# Tiny external health monitor for the live demo.
#
# Hits the /health endpoint and writes a one-line log entry; meant to
# run from cron at a 5-minute cadence during the demo period:
#
#   */5 * * * *  /home/__USER__/ktp-conductor-demo/scripts/healthcheck.sh
#
# The endpoint returns 200 with a JSON body when the server is up and
# the model is loaded. We log the elapsed milliseconds, the HTTP code,
# and (on failure) the response body so the journal contains enough
# detail to diagnose without re-running curl by hand.
#
# Override defaults via environment:
#   URL=https://ktp.example.dev/health LOG=/tmp/ktp.log healthcheck.sh

set -euo pipefail

URL="${URL:-http://localhost:8000/health}"
LOG="${LOG:-${HOME}/ktp-healthcheck.log}"
TIMEOUT="${TIMEOUT:-10}"

mkdir -p "$(dirname "$LOG")"

start_ns=$(date +%s%N)
http_code=$(curl -sS -o /tmp/ktp_health.body -w "%{http_code}" \
    --max-time "$TIMEOUT" "$URL" || echo "000")
elapsed_ms=$(( ( $(date +%s%N) - start_ns ) / 1000000 ))

ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
if [[ "$http_code" == "200" ]]; then
    echo "$ts  ok    $http_code  ${elapsed_ms}ms  $URL" >> "$LOG"
    exit 0
else
    body=$(head -c 256 /tmp/ktp_health.body 2>/dev/null || echo "")
    echo "$ts  FAIL  $http_code  ${elapsed_ms}ms  $URL  body='$body'" >> "$LOG"
    exit 1
fi
