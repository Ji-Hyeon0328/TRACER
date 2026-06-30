#!/usr/bin/env bash
set -euo pipefail

PROFILE="${TRACER_SAFE_BANK_PROFILE:-balanced}"
BANK="${TRACER_SAFE_BANK_PATH:-artifacts/theta_safe_bank_v1/theta_safe_bank_v1.jsonl}"
HZ="${TRACER_SAFE_BANK_HZ:-20.0}"
LOG="${TRACER_SAFE_BANK_LOG:-/tmp/tracer_safe_bank_policy_node_v0.log}"
RESET_DAEMON="${TRACER_RESET_ROS2_DAEMON:-1}"

echo "[TRACER] ensure safe-bank policy node v1"
echo "[TRACER] profile=${PROFILE}"
echo "[TRACER] bank=${BANK}"
echo "[TRACER] hz=${HZ}"
echo "[TRACER] log=${LOG}"
echo "[TRACER] reset_ros2_daemon=${RESET_DAEMON}"

echo
echo "[TRACER] killing old high-level publishers..."
pkill -9 -f 'tracer_safe_bank_policy_node_v0.py' 2>/dev/null || true
pkill -9 -f 'tracer_fusion_policy_mpc_ref_node.py' 2>/dev/null || true
pkill -9 -f 'tracer_objective_selector_stub_node.py' 2>/dev/null || true
pkill -9 -f 'tracer_learned_high_level_policy_udp_client_v1_node.py' 2>/dev/null || true
pkill -9 -f 'tracer_learned_high_level_policy_udp_client_node.py' 2>/dev/null || true
pkill -9 -f 'tracer_high_level_controller_stub_node.py' 2>/dev/null || true
pkill -9 -f 'ros2 topic pub.*/tracer/mpc_reference' 2>/dev/null || true

sleep 1.0

echo
echo "[TRACER] live policy processes after kill:"
pgrep -af 'tracer_safe_bank_policy_node_v0.py' || true

set +u
source /opt/ros/humble/setup.bash
if [ -f ros2_ws/install/setup.bash ]; then
  source ros2_ws/install/setup.bash
fi
set -u

if [ "${RESET_DAEMON}" = "1" ]; then
  echo
  echo "[TRACER] resetting ROS2 daemon to clear stale graph endpoints..."
  ros2 daemon stop >/dev/null 2>&1 || true
  sleep 0.5
  ros2 daemon start >/dev/null 2>&1 || true
  sleep 0.5
fi

echo
echo "[TRACER] starting safe-bank policy node..."
nohup /usr/bin/python3 ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_safe_bank_policy_node_v0.py \
  --ros-args \
  -p safe_bank_path:="${BANK}" \
  -p profile:="${PROFILE}" \
  -p publish_hz:="${HZ}" \
  > "${LOG}" 2>&1 &

sleep 1.5

echo
echo "[TRACER] process check:"
pgrep -af 'tracer_safe_bank_policy_node_v0.py' || {
  echo "[TRACER][ERROR] safe-bank policy node failed to start"
  echo "[TRACER] log tail:"
  tail -80 "${LOG}" || true
  exit 1
}

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
tail -40 "${LOG}" || true
