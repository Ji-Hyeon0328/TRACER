#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
TERRAIN="${1:-${TRACER_TERRAIN_KEY:-flat_normal}}"
DURATION="${2:-${TRACER_PUBLISH_DURATION:-0.0}}"
POLICY_JSON="${TRACER_FUSION_POLICY_JSON:-${TRACER_POLICY_JSON:-$ROOT/configs/highlevel_policy/tracer_fusion_policy_v1.json}}"
LOG="/tmp/tracer_fusion_policy_mpc_ref_${TERRAIN}.log"
PIDFILE="/tmp/tracer_fusion_policy_mpc_ref.pid"

META_POLICY_KIND="${TRACER_META_GAIT_POLICY_KIND:-rule_based}"
META_POLICY_UDP_HOST="${TRACER_META_GAIT_POLICY_UDP_HOST:-127.0.0.1}"
META_POLICY_UDP_PORT="${TRACER_META_GAIT_POLICY_UDP_PORT:-50310}"
ENABLE_GMS="${TRACER_ENABLE_GMS:-1}"
GMS_USE_RAM_GATE="${TRACER_GMS_USE_RAM_GATE:-1}"
GMS_GATE_FRESHNESS_SEC="${TRACER_GMS_GATE_FRESHNESS_SEC:-2.0}"

OBJECTIVE_SELECTOR_KIND="${TRACER_OBJECTIVE_SELECTOR_KIND:-runtime_baseline_json}"
OBJECTIVE_SELECTOR_APPLY_SEMANTIC="${TRACER_OBJECTIVE_SELECTOR_APPLY_SEMANTIC:-1}"
OBJECTIVE_SELECTOR_APPLY_BETA="${TRACER_OBJECTIVE_SELECTOR_APPLY_BETA:-1}"
OBJECTIVE_SELECTOR_BLOCK_NO_DEPLOY="${TRACER_OBJECTIVE_SELECTOR_BLOCK_NO_DEPLOY:-1}"
OBJECTIVE_SELECTOR_VERBOSE="${TRACER_OBJECTIVE_SELECTOR_VERBOSE:-0}"

echo "[TRACER] starting fusion policy overlay from qwerty"
echo "[TRACER] terrain:     $TERRAIN"
echo "[TRACER] duration:    $DURATION"
echo "[TRACER] policy_json: $POLICY_JSON"
echo "[TRACER] meta_policy: $META_POLICY_KIND"
echo "[TRACER] meta_udp:    $META_POLICY_UDP_HOST:$META_POLICY_UDP_PORT"
echo "[TRACER] gms:         enable=$ENABLE_GMS use_ram_gate=$GMS_USE_RAM_GATE freshness=${GMS_GATE_FRESHNESS_SEC}s"
echo "[TRACER] objective:   kind=$OBJECTIVE_SELECTOR_KIND semantic=$OBJECTIVE_SELECTOR_APPLY_SEMANTIC beta=$OBJECTIVE_SELECTOR_APPLY_BETA block=$OBJECTIVE_SELECTOR_BLOCK_NO_DEPLOY verbose=$OBJECTIVE_SELECTOR_VERBOSE"
echo "[TRACER] log:         $LOG"

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
TRACER_FUSION_POLICY_JSON="$POLICY_JSON" \
TRACER_ENABLE_GMS="$ENABLE_GMS" \
TRACER_GMS_USE_RAM_GATE="$GMS_USE_RAM_GATE" \
TRACER_GMS_GATE_FRESHNESS_SEC="$GMS_GATE_FRESHNESS_SEC" \
TRACER_OBJECTIVE_SELECTOR_KIND="$OBJECTIVE_SELECTOR_KIND" \
TRACER_OBJECTIVE_SELECTOR_APPLY_SEMANTIC="$OBJECTIVE_SELECTOR_APPLY_SEMANTIC" \
TRACER_OBJECTIVE_SELECTOR_APPLY_BETA="$OBJECTIVE_SELECTOR_APPLY_BETA" \
TRACER_OBJECTIVE_SELECTOR_BLOCK_NO_DEPLOY="$OBJECTIVE_SELECTOR_BLOCK_NO_DEPLOY" \
TRACER_OBJECTIVE_SELECTOR_VERBOSE="$OBJECTIVE_SELECTOR_VERBOSE" \
TRACER_OBJECTIVE_CONDITIONED_SELECTOR_ENABLE="${TRACER_OBJECTIVE_CONDITIONED_SELECTOR_ENABLE:-0}" \
TRACER_OBJECTIVE_CONDITIONED_SELECTOR_APPLY="${TRACER_OBJECTIVE_CONDITIONED_SELECTOR_APPLY:-0}" \
TRACER_OBJECTIVE_CONDITIONED_SELECTOR_MIN_CONFIDENCE="${TRACER_OBJECTIVE_CONDITIONED_SELECTOR_MIN_CONFIDENCE:-0.05}" \
TRACER_OBJECTIVE_CONDITIONED_SELECTOR_MODEL="${TRACER_OBJECTIVE_CONDITIONED_SELECTOR_MODEL:-configs/highlevel_policy/tracer_objective_conditioned_selector_v0.json}" \
"$ROOT/scripts/runtime/tracer_start_fusion_policy_mpc_ref_ros2.sh" "$TERRAIN" \
  > "$LOG" 2>&1 &

PID="$!"
echo "$PID" > "$PIDFILE"

echo "[TRACER] fusion policy publisher started pid=$PID"
echo "[TRACER] tail log:"
sleep 1.0
tail -n 20 "$LOG" || true
