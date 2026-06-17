#!/usr/bin/env bash
set -euo pipefail

cd ~/Tracer/TRACER/ros2_ws

source /opt/ros/humble/setup.bash
source install/setup.bash 2>/dev/null || true

PY=/usr/bin/python3
SCRIPT_DIR=src/tracer_a1_qpmc_adapter/scripts

echo "[TRACER] Starting ROS2 bridge v1"
echo "[TRACER] 1. UDP odom -> /tracer/robot_odom_flat"
echo "[TRACER] 2. /tracer/global_goal + odom -> /tracer/relative_goal"
echo "[TRACER] 3. /tracer/relative_goal -> /tracer/mpc_reference"
echo "[TRACER] 4. /tracer/mpc_reference -> UDP command"

cleanup() {
  echo "[TRACER] stopping ROS2 bridge v1..."
  kill ${ODOM_RX_PID:-} ${GLOBAL_PID:-} ${GOAL_PID:-} ${CMD_TX_PID:-} 2>/dev/null || true
}
trap cleanup EXIT INT TERM

$PY $SCRIPT_DIR/tracer_udp_odom_to_ros2_node.py &
ODOM_RX_PID=$!

sleep 0.5

$PY $SCRIPT_DIR/tracer_global_goal_to_relative_goal_node.py &
GLOBAL_PID=$!

sleep 0.5

$PY $SCRIPT_DIR/tracer_goal_to_mpc_ref_node.py &
GOAL_PID=$!

sleep 0.5

$PY $SCRIPT_DIR/tracer_ros2_mpc_ref_udp_sender.py &
CMD_TX_PID=$!

wait
