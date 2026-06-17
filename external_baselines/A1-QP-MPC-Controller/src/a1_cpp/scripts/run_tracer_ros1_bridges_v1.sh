#!/usr/bin/env bash
set -e

ODOM_TOPIC="${ODOM_TOPIC:-/torso_odom}"

source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash

echo "[TRACER] Starting ROS1 bridges"
echo "[TRACER] command UDP receiver: UDP 50110 -> /tracer/mpc_reference"
echo "[TRACER] odom UDP sender: ${ODOM_TOPIC} -> UDP 50120"

cleanup() {
  echo "[TRACER] stopping ROS1 bridges..."
  kill ${CMD_PID:-} ${ODOM_PID:-} 2>/dev/null || true
}
trap cleanup EXIT INT TERM

rosrun a1_cpp tracer_udp_to_ros1_mpc_ref.py &
CMD_PID=$!

sleep 0.5

rosrun a1_cpp tracer_ros1_odom_udp_sender.py _odom_topic:="${ODOM_TOPIC}" &
ODOM_PID=$!

wait
