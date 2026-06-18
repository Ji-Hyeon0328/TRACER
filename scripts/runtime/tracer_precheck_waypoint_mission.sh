#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"

echo "[TRACER] delegating to live waypoint mission precheck"
exec "$ROOT/scripts/runtime/tracer_precheck_waypoint_mission_live.sh" "$@"
