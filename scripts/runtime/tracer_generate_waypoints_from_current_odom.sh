#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WS="$ROOT/ros2_ws"

DISTANCES="${1:-1.0,2.0,3.0}"
PUBLISH_DURATION="${2:-5.0}"
PULSE_SEC="${TRACER_ODOM_PULSE_SEC:-1.5}"
TIMEOUT_SEC="${TRACER_WP_TIMEOUT_SEC:-12}"

LOG_DIR="$ROOT/logs/waypoint_generate_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/waypoints_ahead_publisher.log"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true

echo "[TRACER] generate waypoints from current odom"
echo "[TRACER] distances: $DISTANCES"
echo "[TRACER] log: $LOG"

pkill -9 -f tracer_waypoints_ahead_publisher_node.py || true

timeout "$TIMEOUT_SEC" /usr/bin/python3 "$WS/src/tracer_a1_qpmc_adapter/scripts/tracer_waypoints_ahead_publisher_node.py" \
  --ros-args \
  -p distances_csv:="$DISTANCES" \
  -p publish_hz:=2.0 \
  -p publish_duration_sec:="$PUBLISH_DURATION" \
  > "$LOG" 2>&1 &

WP_PID=$!

sleep 0.5

echo "[TRACER] pulsing Gazebo physics for odom stream: ${PULSE_SEC}s"

sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/unpause_physics "{}"
' >/dev/null || true

sleep "$PULSE_SEC"

sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/pause_physics "{}"
' >/dev/null || true

wait "$WP_PID" || true

echo "[TRACER] waypoint publisher log:"
echo "----------------------------------------"
cat "$LOG"
echo "----------------------------------------"

echo "[TRACER] waypoint generation command finished."
