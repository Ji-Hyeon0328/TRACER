#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
TERRAIN="${1:-${TRACER_TERRAIN_KEY:-flat_normal}}"
DURATION="${2:-${TRACER_PUBLISH_DURATION:-0.0}}"
LOG="/tmp/tracer_fusion_policy_mpc_ref_${TERRAIN}.log"
PIDFILE="/tmp/tracer_fusion_policy_mpc_ref.pid"

echo "[TRACER] starting fusion policy overlay from qwerty"
echo "[TRACER] terrain:  $TERRAIN"
echo "[TRACER] duration: $DURATION"
echo "[TRACER] log:      $LOG"

# Stop previous fusion policy publisher only.
if [ -f "$PIDFILE" ]; then
  OLD_PID="$(cat "$PIDFILE" || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[TRACER] stopping previous fusion policy publisher pid=$OLD_PID"
    kill "$OLD_PID" 2>/dev/null || true
    sleep 0.5
  fi
fi

pkill -f "tracer_fusion_policy_mpc_ref_node.py" 2>/dev/null || true

# Stop objective stub because fusion policy will publish /tracer/objective_weights.
pkill -f "tracer_objective_selector_stub_node.py" 2>/dev/null || true

# Stop stale high-level publishers if helper scripts exist.
if [ -x "$ROOT/scripts/runtime/tracer_stop_learned_mpc_ref_path.sh" ]; then
  "$ROOT/scripts/runtime/tracer_stop_learned_mpc_ref_path.sh" || true
fi

if [ -x "$ROOT/scripts/runtime/tracer_cleanup_stale_style_publishers.sh" ]; then
  "$ROOT/scripts/runtime/tracer_cleanup_stale_style_publishers.sh" || true
fi

# Ensure ROS2->UDP and UDP->ROS1 bridge side is alive if this helper exists.
if [ -x "$ROOT/scripts/runtime/tracer_ensure_mpc_ref_bridge.sh" ]; then
  "$ROOT/scripts/runtime/tracer_ensure_mpc_ref_bridge.sh" || true
fi

if [ -x "$ROOT/scripts/runtime/tracer_ensure_ros2_mpc_ref_udp_sender.sh" ]; then
  "$ROOT/scripts/runtime/tracer_ensure_ros2_mpc_ref_udp_sender.sh" || true
fi

TRACER_PUBLISH_DURATION="$DURATION" \
"$ROOT/scripts/runtime/tracer_start_fusion_policy_mpc_ref_ros2.sh" "$TERRAIN" \
  > "$LOG" 2>&1 &

PID="$!"
echo "$PID" > "$PIDFILE"

echo "[TRACER] fusion policy publisher started pid=$PID"
echo "[TRACER] tail log:"
sleep 1.0
tail -n 20 "$LOG" || true
