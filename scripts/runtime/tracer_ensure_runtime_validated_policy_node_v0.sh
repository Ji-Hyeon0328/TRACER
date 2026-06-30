#!/usr/bin/env bash
set -euo pipefail

PROFILE="${TRACER_RUNTIME_SELECTOR_PROFILE:-balanced}"
SELECTOR="${TRACER_RUNTIME_SELECTOR_PATH:-configs/runtime/tracer_runtime_validated_theta_selector_v0.json}"
HZ="${TRACER_RUNTIME_SELECTOR_HZ:-20.0}"
RAMP="${TRACER_RUNTIME_SELECTOR_RAMP_SEC:-1.0}"
RAMP_MODE="${TRACER_RUNTIME_SELECTOR_RAMP_MODE:-single}"
POSTURE_RAMP="${TRACER_RUNTIME_SELECTOR_POSTURE_RAMP_SEC:-0.4}"
VELOCITY_RAMP="${TRACER_RUNTIME_SELECTOR_VELOCITY_RAMP_SEC:-0.8}"
LOG="${TRACER_RUNTIME_SELECTOR_LOG:-/tmp/tracer_runtime_validated_policy_node_v0.log}"
RESET_DAEMON="${TRACER_RESET_ROS2_DAEMON:-0}"

echo "[TRACER] ensure runtime-validated policy node v0"
echo "[TRACER] profile=${PROFILE}"
echo "[TRACER] selector=${SELECTOR}"
echo "[TRACER] hz=${HZ}"
echo "[TRACER] ramp_sec=${RAMP}"
echo "[TRACER] ramp_mode=${RAMP_MODE}"
echo "[TRACER] posture_ramp_sec=${POSTURE_RAMP}"
echo "[TRACER] velocity_ramp_sec=${VELOCITY_RAMP}"
echo "[TRACER] log=${LOG}"
echo "[TRACER] reset_ros2_daemon=${RESET_DAEMON}"

echo
echo "[TRACER] killing old high-level publishers..."
pkill -f "tracer_safe_bank_policy_node_v0.py" || true
pkill -f "tracer_runtime_validated_policy_node_v0.py" || true
pkill -f "tracer_learned_high_level_policy_udp_client_v1_node.py" || true
pkill -f "tracer_objective_selector_stub_node.py" || true
pkill -f "ros2 topic pub /tracer/mpc_reference" || true

sleep 0.5

if [ "${RESET_DAEMON}" = "1" ]; then
  echo
  echo "[TRACER] resetting ROS2 daemon..."
  set +u
  source /opt/ros/humble/setup.bash
  set -u
  ros2 daemon stop || true
  ros2 daemon start || true
fi

set +u
source /opt/ros/humble/setup.bash
if [ -f ros2_ws/install/setup.bash ]; then
  source ros2_ws/install/setup.bash
fi
set -u

echo
echo "[TRACER] starting runtime-validated policy node..."
nohup /usr/bin/python3 ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_runtime_validated_policy_node_v0.py \
  --ros-args \
  -p selector_path:="${SELECTOR}" \
  -p profile:="${PROFILE}" \
  -p publish_hz:="${HZ}" \
  -p policy_ramp_sec:="${RAMP}" \
  -p ramp_mode:="${RAMP_MODE}" \
  -p posture_ramp_sec:="${POSTURE_RAMP}" \
  -p velocity_ramp_sec:="${VELOCITY_RAMP}" \
  > "${LOG}" 2>&1 &

sleep 1.0

echo
echo "[TRACER] process check:"
pgrep -af "tracer_runtime_validated_policy_node_v0.py" || true

echo
echo "[TRACER] selection info:"
timeout 3 ros2 topic echo --once /tracer/runtime_theta_selection_info || true

echo
echo "[TRACER] selected theta:"
timeout 3 ros2 topic echo --once /tracer/selected_theta || true

echo
echo "[TRACER] mpc reference:"
timeout 3 ros2 topic echo --once /tracer/mpc_reference || true

echo
echo "[TRACER] topic info:"
ros2 topic info -v /tracer/mpc_reference || true

echo
echo "[TRACER] log tail:"
tail -n 20 "${LOG}" || true
