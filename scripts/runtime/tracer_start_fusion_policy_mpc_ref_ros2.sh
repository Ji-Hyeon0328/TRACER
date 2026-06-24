#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
TERRAIN="${1:-${TRACER_TERRAIN_KEY:-flat_normal}}"
POLICY_JSON="${TRACER_FUSION_POLICY_JSON:-$ROOT/configs/highlevel_policy/tracer_fusion_policy_v0.json}"
HZ="${TRACER_MPC_REF_HZ:-10.0}"
DURATION="${TRACER_PUBLISH_DURATION:-0.0}"

echo "[TRACER] starting fusion policy MPC ref publisher"
echo "[TRACER] root:        $ROOT"
echo "[TRACER] terrain:     $TERRAIN"
echo "[TRACER] policy_json: $POLICY_JSON"
echo "[TRACER] hz:          $HZ"
echo "[TRACER] duration:    $DURATION"

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

/usr/bin/python3 "$ROOT/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_fusion_policy_mpc_ref_node.py"
