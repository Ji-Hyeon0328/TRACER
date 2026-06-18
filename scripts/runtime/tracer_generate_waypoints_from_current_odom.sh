#!/usr/bin/env bash
set -eo pipefail

# Prefer first positional argument, then environment variable, then default.
TRACER_WAYPOINT_DISTANCES="${1:-${TRACER_WAYPOINT_DISTANCES:-1.0,2.0,3.0}}"

export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WS="$ROOT/ros2_ws"

DISTANCES="$TRACER_WAYPOINT_DISTANCES"
PUBLISH_DURATION="${2:-${TRACER_WAYPOINT_PUBLISH_DURATION_SEC:-5.0}}"
PULSE_SEC="${TRACER_ODOM_PULSE_SEC:-1.5}"
TIMEOUT_SEC="${TRACER_WP_TIMEOUT_SEC:-12}"

# New default: do not unpause Gazebo just to get odom.
# Options:
#   gazebo_state: read /gazebo/get_model_state while physics remains paused
#   odom: legacy behavior, briefly unpause physics to get /tracer/robot_odom_flat
WAYPOINT_SOURCE="${TRACER_WAYPOINT_SOURCE:-gazebo_state}"

GAZEBO_CONTAINER="${TRACER_GAZEBO_CONTAINER:-a1_unitree_gazebo_docker}"
MODEL_NAME="${TRACER_GAZEBO_MODEL_NAME:-a1_gazebo}"

LOG_DIR="$ROOT/logs/waypoint_generate_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/waypoints_ahead_publisher.log"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true

echo "[TRACER] generate waypoints"
echo "[TRACER] waypoint source: $WAYPOINT_SOURCE"
echo "[TRACER] distances: $DISTANCES"
echo "[TRACER] publish duration: $PUBLISH_DURATION"
echo "[TRACER] log: $LOG"

pkill -9 -f tracer_waypoints_ahead_publisher_node.py || true

if [ "$WAYPOINT_SOURCE" = "gazebo_state" ] || [ "$WAYPOINT_SOURCE" = "fixed_pose" ]; then
  echo "[TRACER] reading Gazebo model pose while physics remains paused"
  echo "[TRACER] container: $GAZEBO_CONTAINER"
  echo "[TRACER] model:     $MODEL_NAME"

  POSE_CSV="$(
    sudo docker exec "$GAZEBO_CONTAINER" bash --noprofile --norc -lc "
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash

MODEL_NAME='$MODEL_NAME' python - <<'PY'
import math
import os
import sys
import rospy
import tf.transformations as tft
from gazebo_msgs.srv import GetModelState

model_name = os.environ.get('MODEL_NAME', 'a1_gazebo')

rospy.init_node('tracer_get_model_pose_for_waypoints', anonymous=True, disable_signals=True)

try:
    rospy.wait_for_service('/gazebo/get_model_state', timeout=3.0)
    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    res = get_state(model_name, 'world')
except Exception as e:
    sys.stderr.write('[ERROR] failed to call /gazebo/get_model_state: {}\\n'.format(e))
    sys.exit(2)

if not res.success:
    sys.stderr.write('[ERROR] get_model_state failed: {}\\n'.format(res.status_message))
    sys.exit(3)

p = res.pose.position
q = res.pose.orientation
roll, pitch, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])

print('{:.9f},{:.9f},{:.9f}'.format(p.x, p.y, yaw))
PY
"
  )"

  IFS=',' read -r INIT_X INIT_Y INIT_YAW <<< "$POSE_CSV"

  echo "[TRACER] Gazebo pose for waypoints:"
  echo "  x:   $INIT_X"
  echo "  y:   $INIT_Y"
  echo "  yaw: $INIT_YAW"

  timeout "$TIMEOUT_SEC" /usr/bin/python3 "$WS/src/tracer_a1_qpmc_adapter/scripts/tracer_waypoints_ahead_publisher_node.py" \
    --ros-args \
    -p source_mode:="gazebo_state" \
    -p initial_x:="$INIT_X" \
    -p initial_y:="$INIT_Y" \
    -p initial_yaw:="$INIT_YAW" \
    -p distances_csv:="$DISTANCES" \
    -p publish_hz:=2.0 \
    -p publish_duration_sec:="$PUBLISH_DURATION" \
    > "$LOG" 2>&1 || true

else
  echo "[TRACER] legacy odom mode: pulsing Gazebo physics for odom stream: ${PULSE_SEC}s"

  timeout "$TIMEOUT_SEC" /usr/bin/python3 "$WS/src/tracer_a1_qpmc_adapter/scripts/tracer_waypoints_ahead_publisher_node.py" \
    --ros-args \
    -p source_mode:="odom" \
    -p distances_csv:="$DISTANCES" \
    -p publish_hz:=2.0 \
    -p publish_duration_sec:="$PUBLISH_DURATION" \
    > "$LOG" 2>&1 &

  WP_PID=$!

  sleep 0.5

  sudo docker exec "$GAZEBO_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/unpause_physics "{}"
' >/dev/null || true

  sleep "$PULSE_SEC"

  sudo docker exec "$GAZEBO_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/pause_physics "{}"
' >/dev/null || true

  wait "$WP_PID" || true
fi

echo "[TRACER] waypoint publisher log:"
echo "----------------------------------------"
cat "$LOG"
echo "----------------------------------------"

echo "[TRACER] waypoint generation command finished."
