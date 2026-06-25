#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
PYTHON_BIN="${TRACER_RAM_PYTHON:-$HOME/anaconda3/envs/env_isaaclab/bin/python}"
LOG="${TRACER_RAM_LOG:-/tmp/tracer_ram_udp_server.log}"
PIDFILE="${TRACER_RAM_PIDFILE:-/tmp/tracer_ram_udp_server.pid}"
ARTIFACT="${TRACER_RAM_ARTIFACT:-$ROOT/artifacts/tracer_ram_v0}"

echo "[TRACER] starting RAM UDP server"
echo "[TRACER] root:     $ROOT"
echo "[TRACER] artifact: $ARTIFACT"
echo "[TRACER] python:   $PYTHON_BIN"
echo "[TRACER] log:      $LOG"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[TRACER][ERROR] python not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -e "$ARTIFACT" ]; then
  echo "[TRACER][WARN] RAM artifact path does not exist yet: $ARTIFACT" >&2
fi

if [ -f "$PIDFILE" ]; then
  OLD_PID="$(cat "$PIDFILE" || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[TRACER] stopping previous RAM UDP server pid=$OLD_PID"
    kill "$OLD_PID" 2>/dev/null || true
    sleep 0.5
  fi
fi

pkill -f "tracer_ram_udp_server_v0.py" 2>/dev/null || true

cd "$ROOT"

PYTHONPATH="$ROOT" \
"$PYTHON_BIN" "$ROOT/scripts/runtime/tracer_ram_udp_server_v0.py" \
  > "$LOG" 2>&1 &

PID="$!"
echo "$PID" > "$PIDFILE"

echo "[TRACER] RAM UDP server started pid=$PID"
sleep 1.0
tail -n 20 "$LOG" || true
