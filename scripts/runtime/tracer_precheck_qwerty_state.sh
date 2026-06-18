#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"

echo "[TRACER] delegating to live qwerty precheck"
exec "$ROOT/scripts/runtime/tracer_precheck_qwerty_state_live.sh" "$@"
