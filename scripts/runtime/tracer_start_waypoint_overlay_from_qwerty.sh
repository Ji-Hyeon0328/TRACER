#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WS="$ROOT/ros2_ws"
LOG_DIR="$ROOT/logs/waypoint_overlay_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "[TRACER] start waypoint overlay from qwerty"
echo "[TRACER] log dir: $LOG_DIR"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true

echo
echo "[TRACER] cleaning previous waypoint/odom overlay nodes on host..."
pkill -9 -f tracer_udp_odom_to_ros2_node.py || true
pkill -9 -f tracer_global_goal_to_relative_goal_node.py || true
pkill -9 -f tracer_global_goal_to_relative_goal_v1_node.py || true
pkill -9 -f tracer_waypoint_manager_node.py || true
pkill -9 -f tracer_waypoint_manager_v1_node.py || true
pkill -9 -f tracer_waypoints_ahead_publisher_node.py || true

echo
echo "[TRACER] cleaning previous ROS1 odom sender inside a1_cpp_ctrl_docker..."
sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
pkill -9 -f tracer_ros1_odom_udp_sender.py || true
' || true

echo
echo "[TRACER] starting ROS1 odom sender inside a1_cpp_ctrl_docker..."
sudo docker exec -d a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rosrun a1_cpp tracer_ros1_odom_udp_sender.py _odom_topic:=/torso_odom > /tmp/tracer_ros1_odom_udp_sender.log 2>&1
'

start_host_node () {
  local name="$1"
  shift

  echo "[TRACER] starting host node: $name"
  nohup "$@" > "$LOG_DIR/${name}.log" 2>&1 &
  echo $! > "$LOG_DIR/${name}.pid"
  sleep 0.4
}

echo
echo "[TRACER] starting ROS2 odom receiver..."
start_host_node "ros2_odom_receiver" \
  /usr/bin/python3 "$WS/src/tracer_a1_qpmc_adapter/scripts/tracer_udp_odom_to_ros2_node.py"

echo
echo "[TRACER] starting global-to-relative v1..."
start_host_node "global_goal_to_relative_goal_v1" \
  /usr/bin/python3 "$WS/src/tracer_a1_qpmc_adapter/scripts/tracer_global_goal_to_relative_goal_v1_node.py" \
  --ros-args \
  -p odom_timeout_sec:=30.0 \
  -p goal_timeout_sec:=30.0 \
  -p stop_radius:=0.15

echo
echo "[TRACER] starting waypoint manager v1..."
start_host_node "waypoint_manager_v1" \
  /usr/bin/python3 "$WS/src/tracer_a1_qpmc_adapter/scripts/tracer_waypoint_manager_v1_node.py" \
  --ros-args \
  -p reach_radius:=0.55 \
  -p pass_radius:=0.80 \
  -p pass_hysteresis:=0.25

echo
echo "[TRACER] waypoint overlay started from qwerty state."
echo "[TRACER] next:"
echo "  scripts/runtime/tracer_generate_waypoints_from_current_odom.sh"
echo "  scripts/runtime/tracer_precheck_waypoint_mission.sh"
echo "  scripts/runtime/tracer_unpause_mission.sh"
echo
echo "[TRACER] logs: $LOG_DIR"
