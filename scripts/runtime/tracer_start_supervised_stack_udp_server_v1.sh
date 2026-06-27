#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
HOST="${TRACER_SUPERVISED_STACK_HOST:-127.0.0.1}"
PORT="${TRACER_SUPERVISED_STACK_PORT:-50410}"
LOG="${TRACER_SUPERVISED_STACK_LOG:-/tmp/tracer_supervised_stack_udp_server_v1.log}"

cd "$ROOT"

echo "[TRACER] starting supervised high-level stack UDP server v1"
echo "[TRACER] root: $ROOT"
echo "[TRACER] bind: $HOST:$PORT"
echo "[TRACER] log:  $LOG"

pkill -f "tracer_supervised_stack_udp_server_v1.py" || true
sleep 0.5

python3 -u scripts/runtime/tracer_supervised_stack_udp_server_v1.py \
  --host "$HOST" \
  --port "$PORT" \
  > "$LOG" 2>&1 &

PID=$!
echo "[TRACER] supervised stack server pid=$PID"

sleep 2.0

echo "[TRACER] process check:"
pgrep -af "tracer_supervised_stack_udp_server_v1.py" || true

echo "[TRACER] UDP check:"
ss -lunp | grep ":$PORT" || true

echo "[TRACER] log tail:"
tail -n 40 "$LOG" || true
