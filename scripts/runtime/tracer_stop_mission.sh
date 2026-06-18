#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WS="$ROOT/ros2_ws"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true

echo "[TRACER] publish mission stop"
ros2 topic pub --once /tracer/mission_cmd std_msgs/msg/Float64MultiArray "{data: [0.0]}" || true
ros2 topic pub --once /tracer/global_goal std_msgs/msg/Float64MultiArray "{data: [0.0, 0.0, 0.0]}" || true

echo "[TRACER] pause Gazebo physics"
sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/pause_physics "{}"
' || true

echo "[TRACER] stopped."
