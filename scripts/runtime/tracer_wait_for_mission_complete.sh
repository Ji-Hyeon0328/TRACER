#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
TIMEOUT_SEC="${TRACER_MISSION_TIMEOUT_SEC:-90}"

if [ $# -ge 1 ]; then
  LOG_FILE="$1"
else
  LOG_FILE="$(ls -td "$ROOT"/logs/waypoint_overlay_* 2>/dev/null | head -1)/waypoint_manager_v1.log"
fi

echo "[TRACER] waiting for mission complete"
echo "[TRACER] log file: $LOG_FILE"
echo "[TRACER] timeout:  $TIMEOUT_SEC sec"

if [ ! -f "$LOG_FILE" ]; then
  echo "[TRACER] ERROR: waypoint manager log not found."
  exit 2
fi

if timeout "$TIMEOUT_SEC" bash -c 'tail -n 50 -F "$0" | grep -m 1 "mission complete"' "$LOG_FILE"; then
  echo "[TRACER] mission complete detected."
  exit 0
else
  echo "[TRACER] mission complete not detected within timeout."
  exit 1
fi
