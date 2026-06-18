#!/usr/bin/env bash
set -eo pipefail

MAX_TRIES="${TRACER_RESET_MAX_TRIES:-3}"

echo "[TRACER] robust reset to qwer state"
echo "[TRACER] max tries: $MAX_TRIES"

for i in $(seq 1 "$MAX_TRIES"); do
  echo
  echo "============================================================"
  echo "[TRACER] robust reset attempt $i / $MAX_TRIES"
  echo "============================================================"

  scripts/runtime/tracer_cleanup_stale_style_publishers.sh || true
  scripts/runtime/tracer_stop_a1_qpmc_controller_only.sh || true

  # Hard pose reset first, because servo reset may not recover from fallen pose.
  scripts/runtime/tracer_hard_reset_a1_pose.sh || true

  # Existing standing-pose sequence.
  scripts/runtime/tracer_reset_to_qwer_state.sh || true

  # Validate final pose.
  if scripts/runtime/tracer_validate_a1_initial_pose.sh; then
    echo "[TRACER] robust reset succeeded on attempt $i"
    exit 0
  fi

  echo "[TRACER] robust reset validation failed on attempt $i"
  sleep 1
done

echo "[ERROR] robust reset failed after $MAX_TRIES attempts"
exit 30
