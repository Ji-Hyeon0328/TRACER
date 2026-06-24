#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
LOG="/tmp/tracer_ros2_mpc_ref_udp_sender.log"
PIDFILE="/tmp/tracer_ros2_mpc_ref_udp_sender.pid"

echo "[TRACER] ensure ROS2 MPC reference UDP sender"

if pgrep -af "tracer_ros2_mpc_ref_udp_sender.py" >/dev/null 2>&1; then
  echo "[TRACER] existing ROS2 MPC ref UDP sender:"
  pgrep -af "tracer_ros2_mpc_ref_udp_sender.py" || true
  exit 0
fi

set +u
source /opt/ros/humble/setup.bash
if [ -f "$ROOT/ros2_ws/install/setup.bash" ]; then
  source "$ROOT/ros2_ws/install/setup.bash"
fi
set -u

echo "[TRACER] starting tracer_ros2_mpc_ref_udp_sender.py"
echo "[TRACER] log: $LOG"

/usr/bin/python3 "$ROOT/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_ros2_mpc_ref_udp_sender.py" \
  > "$LOG" 2>&1 &

PID="$!"
echo "$PID" > "$PIDFILE"

sleep 1.0

echo "[TRACER] process check:"
pgrep -af "tracer_ros2_mpc_ref_udp_sender.py" || true

echo "[TRACER] log tail:"
tail -n 20 "$LOG" || true
