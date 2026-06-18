#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
PID_FILE="$ROOT/logs/latest_mission_logger.pid"

echo "[TRACER] stopping mission logger"

if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "[TRACER] sending SIGINT to logger pid=$PID"
    kill -INT "$PID" 2>/dev/null || true

    for _ in $(seq 1 30); do
      if ! kill -0 "$PID" 2>/dev/null; then
        break
      fi
      sleep 0.2
    done

    if kill -0 "$PID" 2>/dev/null; then
      echo "[TRACER] logger still alive; sending SIGTERM"
      kill -TERM "$PID" 2>/dev/null || true
      sleep 0.5
    fi
  fi
else
  echo "[TRACER] pid file not found; fallback pkill"
  pkill -INT -f tracer_mission_logger_node.py || true
  sleep 0.5
fi

echo
echo "[TRACER] latest mission log files:"
ls -t "$ROOT"/data/mission_logs/tracer_mission_* 2>/dev/null | head -5 || true

echo
echo "[TRACER] latest logger stdout:"
if [ -L "$ROOT/logs/latest_mission_logger" ]; then
  tail -n 30 "$ROOT/logs/latest_mission_logger/mission_logger.log" 2>/dev/null || true
fi
