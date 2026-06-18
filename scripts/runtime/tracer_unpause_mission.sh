#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

echo "[TRACER] unpause Gazebo physics"

sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/unpause_physics "{}"
'
