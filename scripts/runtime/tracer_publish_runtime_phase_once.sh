#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-policy}"

echo "[TRACER] publish runtime phase: ${PHASE}"

set +u
source /opt/ros/humble/setup.bash
if [ -f ros2_ws/install/setup.bash ]; then
  source ros2_ws/install/setup.bash
fi
set -u

ros2 topic pub --once /tracer/runtime_phase std_msgs/msg/String "{data: '${PHASE}'}"

echo
echo "[TRACER] current mpc ref:"
timeout 3 ros2 topic echo --once /tracer/mpc_reference || true
