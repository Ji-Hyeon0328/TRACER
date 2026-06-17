#!/usr/bin/env bash
set -euo pipefail

X="${1:-2.0}"
Y="${2:-0.0}"
ENABLE="${3:-1.0}"
RATE="${4:-1}"

cd ~/Tracer/TRACER/ros2_ws

source /opt/ros/humble/setup.bash
source install/setup.bash 2>/dev/null || true

echo "[TRACER] publishing /tracer/global_goal: x=${X}, y=${Y}, enable=${ENABLE}, rate=${RATE}Hz"

ros2 topic pub -r "${RATE}" /tracer/global_goal std_msgs/msg/Float64MultiArray \
"{data: [${X}, ${Y}, ${ENABLE}]}"
