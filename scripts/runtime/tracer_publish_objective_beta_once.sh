#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-balanced}"

case "${PROFILE}" in
  balanced)
    BETA="[0.34, 0.33, 0.33]"
    ;;
  motion)
    BETA="[0.65, 0.20, 0.15]"
    ;;
  stability)
    BETA="[0.20, 0.65, 0.15]"
    ;;
  energy)
    BETA="[0.20, 0.20, 0.60]"
    ;;
  motion_extreme)
    BETA="[0.85, 0.10, 0.05]"
    ;;
  stability_extreme)
    BETA="[0.05, 0.90, 0.05]"
    ;;
  energy_extreme)
    BETA="[0.05, 0.10, 0.85]"
    ;;
  *)
    echo "[TRACER][ERROR] unknown profile: ${PROFILE}"
    echo "valid: balanced motion stability energy motion_extreme stability_extreme energy_extreme"
    exit 1
    ;;
esac

echo "[TRACER] publish beta profile=${PROFILE} beta=${BETA}"

set +u
source /opt/ros/humble/setup.bash
if [ -f ros2_ws/install/setup.bash ]; then
  source ros2_ws/install/setup.bash
fi
set -u

ros2 topic pub --once /tracer/objective_beta std_msgs/msg/Float64MultiArray \
"{data: ${BETA}}"

echo
echo "[TRACER] selected theta after beta:"
timeout 3 ros2 topic echo --once /tracer/selected_theta || true

echo
echo "[TRACER] mpc ref after beta:"
timeout 3 ros2 topic echo --once /tracer/mpc_reference || true
