#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WS="$ROOT/ros2_ws"
PULSE_SEC="${TRACER_WAYPOINT_PRECHECK_PULSE_SEC:-3.0}"

LOG_DIR="$ROOT/logs/waypoint_live_precheck_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true

echo
echo "========== waypoint mission node sanity =========="
echo "[waypoint nodes]"
ros2 node list | grep waypoint || true
echo
echo "[goal/odom nodes]"
ros2 node list | grep -E "odom|goal" || true
echo

echo "========== start topic probes BEFORE physics pulse =========="
echo "[TRACER] logs: $LOG_DIR"

timeout 8 ros2 topic echo /tracer/robot_odom_flat --once > "$LOG_DIR/robot_odom_flat.txt" 2>&1 &
PID_ODOM=$!

timeout 8 ros2 topic echo /tracer/global_goal --once > "$LOG_DIR/global_goal.txt" 2>&1 &
PID_GLOBAL=$!

timeout 8 ros2 topic echo /tracer/relative_goal --once > "$LOG_DIR/relative_goal.txt" 2>&1 &
PID_REL=$!

timeout 8 ros2 topic echo /tracer/context_vector --once > "$LOG_DIR/context_vector.txt" 2>&1 &
PID_CTX=$!

timeout 8 ros2 topic echo /tracer/latent_vector --once > "$LOG_DIR/latent_vector.txt" 2>&1 &
PID_LAT=$!

timeout 8 ros2 topic echo /tracer/mpc_reference --once > "$LOG_DIR/mpc_reference.txt" 2>&1 &
PID_MPC=$!

sleep 0.5

echo
echo "========== pulse Gazebo physics while probes are active =========="
echo "[TRACER] unpause for ${PULSE_SEC}s"

sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/unpause_physics "{}"
' >/dev/null || true

sleep "$PULSE_SEC"

echo "[TRACER] pause again"

sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/pause_physics "{}"
' >/dev/null || true

wait "$PID_ODOM" || true
wait "$PID_GLOBAL" || true
wait "$PID_REL" || true
wait "$PID_CTX" || true
wait "$PID_LAT" || true
wait "$PID_MPC" || true

show_file () {
  local label="$1"
  local file="$2"
  echo
  echo "[$label]"
  if [ -s "$file" ]; then
    cat "$file"
  else
    echo "<empty>"
  fi
}

echo
echo "========== waypoint live topic check =========="
show_file "/tracer/robot_odom_flat" "$LOG_DIR/robot_odom_flat.txt"
show_file "/tracer/global_goal" "$LOG_DIR/global_goal.txt"
show_file "/tracer/relative_goal" "$LOG_DIR/relative_goal.txt"
show_file "/tracer/context_vector" "$LOG_DIR/context_vector.txt"
show_file "/tracer/latent_vector" "$LOG_DIR/latent_vector.txt"
show_file "/tracer/mpc_reference" "$LOG_DIR/mpc_reference.txt"

echo
echo "========== expected =========="
echo "global_goal:    [x_goal, y_goal, 1.0]"
echo "relative_goal:  [x_rel, y_rel, yaw_rel, 1.0]"
echo "context/latent: should publish during physics pulse"
echo "mpc_reference:  [counter, vx > 0, yaw_rate, body_height, clearance, 1.0]"
echo
echo "If mpc_reference is still enable=0 while relative_goal/context/latent are alive,"
echo "then inspect learned_high_level_policy_udp_client_v1 logs."
