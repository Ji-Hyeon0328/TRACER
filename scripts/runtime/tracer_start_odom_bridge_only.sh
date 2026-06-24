#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"

ROS1_LOG="/tmp/tracer_ros1_odom_udp_sender.log"
ROS2_LOG="/tmp/tracer_udp_odom_to_ros2_node.log"

echo "[TRACER] start odom bridge only"
echo "[TRACER] root: $ROOT"

echo
echo "========== stop old odom bridge =========="
pkill -f "tracer_udp_odom_to_ros2_node.py" 2>/dev/null || true

sudo docker exec a1_cpp_ctrl_docker bash -lc '
pkill -f tracer_ros1_odom_udp_sender.py 2>/dev/null || true
' || true

sleep 0.5

echo
echo "========== start ROS1 odom UDP sender =========="
sudo docker exec -d a1_cpp_ctrl_docker bash --noprofile --norc -lc "
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rosrun a1_cpp tracer_ros1_odom_udp_sender.py > $ROS1_LOG 2>&1
"

echo
echo "========== start ROS2 odom UDP receiver =========="
set +u
source /opt/ros/humble/setup.bash
set -u

nohup /usr/bin/python3 \
  "$ROOT/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_udp_odom_to_ros2_node.py" \
  > "$ROS2_LOG" 2>&1 &

PID=$!
echo "[TRACER] ROS2 odom receiver pid=$PID"

sleep 1

echo
echo "========== process check =========="
echo "[ROS1 sender]"
sudo docker exec a1_cpp_ctrl_docker bash -lc 'pgrep -af tracer_ros1_odom_udp_sender.py || true'

echo
echo "[ROS2 receiver]"
pgrep -af "tracer_udp_odom_to_ros2_node.py" || true

echo
echo "========== log tails =========="
echo "[ROS1 sender log]"
sudo docker exec a1_cpp_ctrl_docker bash -lc "tail -n 20 $ROS1_LOG 2>/dev/null || true"

echo
echo "[ROS2 receiver log]"
tail -n 20 "$ROS2_LOG" 2>/dev/null || true
