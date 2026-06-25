#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
TERRAIN="${1:-${TRACER_TERRAIN_KEY:-flat_normal}}"
POLICY_JSON="${TRACER_FUSION_POLICY_JSON:-$ROOT/configs/highlevel_policy/tracer_fusion_policy_v0.json}"
HZ="${TRACER_MPC_REF_HZ:-10.0}"
DURATION="${TRACER_PUBLISH_DURATION:-0.0}"
META_POLICY_KIND="${TRACER_META_GAIT_POLICY_KIND:-rule_based}"
META_POLICY_MODEL="${TRACER_META_GAIT_POLICY_MODEL:-}"
META_POLICY_UDP_HOST="${TRACER_META_GAIT_POLICY_UDP_HOST:-127.0.0.1}"
META_POLICY_UDP_PORT="${TRACER_META_GAIT_POLICY_UDP_PORT:-50310}"
META_POLICY_UDP_TIMEOUT="${TRACER_META_GAIT_POLICY_UDP_TIMEOUT_SEC:-0.20}"
ENABLE_GMS="${TRACER_ENABLE_GMS:-1}"
GMS_USE_RAM_GATE="${TRACER_GMS_USE_RAM_GATE:-0}"
GMS_GATE_FRESHNESS="${TRACER_GMS_GATE_FRESHNESS_SEC:-2.0}"

echo "[TRACER] starting fusion policy MPC ref publisher"
echo "[TRACER] root:        $ROOT"
echo "[TRACER] terrain:     $TERRAIN"
echo "[TRACER] policy_json: $POLICY_JSON"
echo "[TRACER] hz:          $HZ"
echo "[TRACER] duration:    $DURATION"
echo "[TRACER] meta_policy: $META_POLICY_KIND"
echo "[TRACER] meta_model:  ${META_POLICY_MODEL:-<none>}"
echo "[TRACER] meta_udp:    $META_POLICY_UDP_HOST:$META_POLICY_UDP_PORT timeout=${META_POLICY_UDP_TIMEOUT}s"
echo "[TRACER] gms:         enable=$ENABLE_GMS use_ram_gate=$GMS_USE_RAM_GATE freshness=${GMS_GATE_FRESHNESS}s"

# ROS setup files may reference unset variables internally.
# Keep nounset disabled while sourcing ROS environments.
set +u
source /opt/ros/humble/setup.bash

if [ -f "$ROOT/ros2_ws/install/setup.bash" ]; then
  source "$ROOT/ros2_ws/install/setup.bash"
fi
set -u

export TRACER_TERRAIN_KEY="$TERRAIN"
export TRACER_FUSION_POLICY_JSON="$POLICY_JSON"
export TRACER_MPC_REF_HZ="$HZ"
export TRACER_PUBLISH_DURATION="$DURATION"
export TRACER_META_GAIT_POLICY_KIND="$META_POLICY_KIND"
export TRACER_META_GAIT_POLICY_MODEL="$META_POLICY_MODEL"
export TRACER_META_GAIT_POLICY_UDP_HOST="$META_POLICY_UDP_HOST"
export TRACER_META_GAIT_POLICY_UDP_PORT="$META_POLICY_UDP_PORT"
export TRACER_META_GAIT_POLICY_UDP_TIMEOUT_SEC="$META_POLICY_UDP_TIMEOUT"
export TRACER_ENABLE_GMS="$ENABLE_GMS"
export TRACER_GMS_USE_RAM_GATE="$GMS_USE_RAM_GATE"
export TRACER_GMS_GATE_FRESHNESS_SEC="$GMS_GATE_FRESHNESS"

/usr/bin/python3 "$ROOT/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_fusion_policy_mpc_ref_node.py"
