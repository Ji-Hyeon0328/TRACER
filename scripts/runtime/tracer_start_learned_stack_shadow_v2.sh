#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
TERRAIN="${1:-${TRACER_TERRAIN:-flat_normal}}"
DURATION="${2:-${TRACER_SHADOW_DURATION_SEC:-0.0}}"
PORT="${TRACER_SUPERVISED_STACK_PORT:-50410}"
HOST="${TRACER_SUPERVISED_STACK_HOST:-127.0.0.1}"
OUT_DIR="${TRACER_SHADOW_OUT_DIR:-$ROOT/data/shadow_logs_v2}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_CSV="${TRACER_SHADOW_OUT_CSV:-$OUT_DIR/learned_stack_shadow_v2_${TERRAIN}_${TS}.csv}"
LOG="${TRACER_SHADOW_LOG:-/tmp/tracer_learned_stack_shadow_v2_${TERRAIN}.log}"

cd "$ROOT"

set +u
source /opt/ros/humble/setup.bash
set -u

mkdir -p "$OUT_DIR"

echo "[TRACER] starting learned stack shadow node v2"
echo "[TRACER] terrain:  $TERRAIN"
echo "[TRACER] duration: $DURATION"
echo "[TRACER] udp:      $HOST:$PORT"
echo "[TRACER] out_csv:  $OUT_CSV"
echo "[TRACER] log:      $LOG"

if ! ss -lunp | grep -q ":$PORT"; then
  echo "[TRACER][WARN] supervised stack UDP server does not appear to listen on :$PORT"
fi

pkill -f "tracer_learned_stack_shadow_node_v2.py" || true
sleep 0.3

TRACER_TERRAIN="$TERRAIN" \
TRACER_SHADOW_TERRAIN="$TERRAIN" \
TRACER_SHADOW_DURATION_SEC="$DURATION" \
TRACER_SUPERVISED_STACK_HOST="$HOST" \
TRACER_SUPERVISED_STACK_PORT="$PORT" \
TRACER_SHADOW_OUT_CSV="$OUT_CSV" \
/usr/bin/python3 ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_learned_stack_shadow_node_v2.py \
  > "$LOG" 2>&1 &

PID=$!
echo "[TRACER] shadow v2 node pid=$PID"

sleep 1.0

echo "[TRACER] process check:"
pgrep -af "tracer_learned_stack_shadow_node_v2.py" || true

echo "[TRACER] log tail:"
tail -n 30 "$LOG" || true
