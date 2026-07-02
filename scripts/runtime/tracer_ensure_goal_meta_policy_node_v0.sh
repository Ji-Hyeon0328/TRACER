#!/usr/bin/env bash
set -euo pipefail

cd ~/Tracer/TRACER

REL_NODE="tracer_global_goal_to_relative_goal_v1_node.py"
POLICY_NODE="tracer_goal_meta_policy_node_v0.py"

SCRIPT_DIR="ros2_ws/src/tracer_a1_qpmc_adapter/scripts"

PUBLISH_HZ="${TRACER_PHASE_B_PUBLISH_HZ:-20.0}"
GOAL_TIMEOUT="${TRACER_PHASE_B_GOAL_TIMEOUT_SEC:-1.0}"
ODOM_TIMEOUT="${TRACER_PHASE_B_ODOM_TIMEOUT_SEC:-1.0}"

VX_FAR="${TRACER_PHASE_B_VX_FAR:-0.09}"
VX_NEAR="${TRACER_PHASE_B_VX_NEAR:-0.04}"
YAW_GAIN="${TRACER_PHASE_B_YAW_GAIN:-1.2}"
YAW_RATE_MAX="${TRACER_PHASE_B_YAW_RATE_MAX:-0.30}"
BODY_HEIGHT="${TRACER_PHASE_B_BODY_HEIGHT:-0.32}"
SWING_CLEARANCE="${TRACER_PHASE_B_SWING_CLEARANCE:-0.04}"
STOP_DISTANCE="${TRACER_PHASE_B_GOAL_STOP_DISTANCE:-0.15}"
SLOW_DISTANCE="${TRACER_PHASE_B_GOAL_SLOW_DISTANCE:-0.25}"

REL_STOP_RADIUS="${TRACER_PHASE_B_RELATIVE_STOP_RADIUS:-0.0}"

echo "[TRACER] stopping conflicting high-level / mpc_reference publishers..."

pkill -f "tracer_goal_to_mpc_ref_node.py" || true
pkill -f "tracer_goal_meta_policy_node_v0.py" || true
pkill -f "tracer_global_goal_to_relative_goal_node.py" || true
pkill -f "tracer_global_goal_to_relative_goal_v1_node.py" || true

pkill -f "tracer_learned_high_level_policy_udp_client_v1_node.py" || true
pkill -f "tracer_learned_high_level_policy_udp_client_node.py" || true
pkill -f "tracer_objective_conditioned_high_level_controller_stub_node.py" || true
pkill -f "tracer_high_level_controller_stub_node.py" || true
pkill -f "tracer_runtime_validated_policy_node_v0.py" || true
pkill -f "tracer_safe_bank_policy_node_v0.py" || true
pkill -f "tracer_phase_a_ram_aware_policy_node_v0.py" || true
pkill -f "tracer_phase_a_override_policy_node_v0.py" || true
pkill -f "tracer_fusion_policy_mpc_ref_node.py" || true
pkill -f "ros2 topic pub /tracer/mpc_reference" || true

set +u
source /opt/ros/humble/setup.bash
if [ -f ~/Tracer/TRACER/ros2_ws/install/setup.bash ]; then
  source ~/Tracer/TRACER/ros2_ws/install/setup.bash
fi
set -u

echo "[TRACER] starting relative goal converter..."
/usr/bin/python3 -u "${SCRIPT_DIR}/${REL_NODE}" \
  --ros-args \
  -p publish_hz:="${PUBLISH_HZ}" \
  -p goal_timeout_sec:="${GOAL_TIMEOUT}" \
  -p odom_timeout_sec:="${ODOM_TIMEOUT}" \
  -p stop_radius:="${REL_STOP_RADIUS}" \
  > /tmp/tracer_global_goal_to_relative_goal_v1_node.log 2>&1 &

sleep 0.5

echo "[TRACER] starting Phase-B goal meta-policy node..."
/usr/bin/python3 -u "${SCRIPT_DIR}/${POLICY_NODE}" \
  --ros-args \
  -p publish_hz:="${PUBLISH_HZ}" \
  -p goal_timeout_sec:="${GOAL_TIMEOUT}" \
  -p odom_timeout_sec:="${ODOM_TIMEOUT}" \
  -p vx_far:="${VX_FAR}" \
  -p vx_near:="${VX_NEAR}" \
  -p yaw_gain:="${YAW_GAIN}" \
  -p yaw_rate_max:="${YAW_RATE_MAX}" \
  -p body_height:="${BODY_HEIGHT}" \
  -p swing_clearance:="${SWING_CLEARANCE}" \
  -p goal_stop_distance:="${STOP_DISTANCE}" \
  -p goal_slow_distance:="${SLOW_DISTANCE}" \
  > /tmp/tracer_goal_meta_policy_node_v0.log 2>&1 &

sleep 1.0

echo "[TRACER] Phase-B nodes launched."
echo "[TRACER] logs:"
echo "  /tmp/tracer_global_goal_to_relative_goal_v1_node.log"
echo "  /tmp/tracer_goal_meta_policy_node_v0.log"

ros2 topic info -v /tracer/mpc_reference || true
