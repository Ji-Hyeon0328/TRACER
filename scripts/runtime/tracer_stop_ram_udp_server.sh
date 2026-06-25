#!/usr/bin/env bash
set -eo pipefail

PIDFILE="${TRACER_RAM_PIDFILE:-/tmp/tracer_ram_udp_server.pid}"

echo "[TRACER] stopping RAM UDP server"

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE" || true)"
  if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
    echo "[TRACER] stopping pid=$PID"
    kill "$PID" 2>/dev/null || true
    sleep 0.5
  fi
  rm -f "$PIDFILE"
fi

pkill -f "tracer_ram_udp_server_v0.py" 2>/dev/null || true
echo "[TRACER] RAM UDP server stopped"
