#!/usr/bin/env bash
set -euo pipefail

PROFILE="${TRACER_OBJECTIVE_PROFILE:-balanced}"
HZ="${TRACER_OBJECTIVE_HZ:-5.0}"
LOG="${TRACER_OBJECTIVE_LOG:-/tmp/tracer_objective_selector_beta_node_v0.log}"
RESET_DAEMON="${TRACER_RESET_ROS2_DAEMON:-0}"

echo "[TRACER] ensure objective selector beta node v0"
echo "[TRACER] profile=${PROFILE}"
echo "[TRACER] hz=${HZ}"
echo "[TRACER] log=${LOG}"
echo "[TRACER] reset_ros2_daemon=${RESET_DAEMON}"

echo
echo "[TRACER] killing old objective selector beta nodes..."
pkill -f "tracer_objective_selector_beta_node_v0.py" || true
pkill -f "tracer_objective_selector_stub_node.py" || true

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
echo "[TRACER] starting objective selector beta node..."
nohup /usr/bin/python3 ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_objective_selector_beta_node_v0.py \
  --ros-args \
  -p profile:="${PROFILE}" \
  -p publish_hz:="${HZ}" \
  > "${LOG}" 2>&1 &

sleep 1.0

echo
echo "[TRACER] process check:"
pgrep -af "tracer_objective_selector_beta_node_v0.py" || true

echo
echo "[TRACER] objective selector info:"
timeout 3 ros2 topic echo --once /tracer/objective_selector_info || true

echo
echo "[TRACER] objective beta:"
timeout 3 ros2 topic echo --once /tracer/objective_beta || true

echo
echo "[TRACER] log tail:"
tail -n 20 "${LOG}" || true
