#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WS="$ROOT/ros2_ws"
CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true

LOG_DIR="$ROOT/logs/mpc_bridge_precheck_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "[TRACER] MPC ref bridge precheck"
echo "[TRACER] log dir: $LOG_DIR"

echo
echo "[TRACER] ensure bridge first..."
"$ROOT/scripts/runtime/tracer_ensure_mpc_ref_bridge.sh"

echo
echo "[TRACER] start ROS1 echo in controller container..."
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
timeout 8 rostopic echo /tracer/mpc_reference -n 1
' > "$LOG_DIR/ros1_mpc_reference_echo.txt" 2>&1 &
ECHO_PID=$!

sleep 1.0

echo
echo "[TRACER] publish one ROS2 test mpc_reference..."
ros2 topic pub --once /tracer/mpc_reference std_msgs/msg/Float64MultiArray \
"{data: [9999.0, 0.22, 0.0, 0.30, 0.035, 1.0]}" >/dev/null || true

wait "$ECHO_PID" || true

echo
echo "========== ROS1 echo result =========="
cat "$LOG_DIR/ros1_mpc_reference_echo.txt" || true

echo
echo "========== expected =========="
echo "ROS1 echo should contain:"
echo "data: [9999.0, 0.22, 0.0, 0.30, 0.035, 1.0]"
