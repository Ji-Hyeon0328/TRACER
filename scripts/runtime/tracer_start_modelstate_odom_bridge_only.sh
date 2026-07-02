#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"

ROS1_LOG="/tmp/tracer_ros1_model_states_odom_udp_sender.log"
ROS2_LOG="/tmp/tracer_udp_odom_to_ros2_node.log"

MODEL_NAME="${TRACER_GAZEBO_MODEL_NAME:-a1_gazebo}"
UDP_PORT="${TRACER_ODOM_UDP_PORT:-50120}"

echo "[TRACER] start model-state odom bridge only"
echo "[TRACER] root: $ROOT"
echo "[TRACER] model: $MODEL_NAME"
echo "[TRACER] udp port: $UDP_PORT"

echo
echo "========== stop old odom bridge =========="
pkill -f "tracer_udp_odom_to_ros2_node.py" 2>/dev/null || true

docker exec a1_cpp_ctrl_docker bash -lc '
pkill -f tracer_ros1_odom_udp_sender.py 2>/dev/null || true
pkill -f tracer_ros1_model_states_odom_udp_sender.py 2>/dev/null || true
' || true

sleep 0.5

echo
echo "========== copy ROS1 model-state odom sender into container =========="
docker cp \
  "$ROOT/scripts/runtime/tracer_ros1_model_states_odom_udp_sender.py" \
  a1_cpp_ctrl_docker:/tmp/tracer_ros1_model_states_odom_udp_sender.py

echo
echo "========== start ROS1 model-state odom UDP sender =========="
docker exec -d a1_cpp_ctrl_docker bash --noprofile --norc -lc "
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash 2>/dev/null || true
source /root/A1_ctrl_ws/devel/setup.bash 2>/dev/null || true
python /tmp/tracer_ros1_model_states_odom_udp_sender.py \
  _model_name:=${MODEL_NAME} \
  _udp_ip:=127.0.0.1 \
  _udp_port:=${UDP_PORT} \
  > ${ROS1_LOG} 2>&1
"

echo
echo "========== start ROS2 odom UDP receiver =========="
set +u
source /opt/ros/humble/setup.bash
set -u

nohup /usr/bin/python3 \
  "$ROOT/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_udp_odom_to_ros2_node.py" \
  --ros-args \
  -p udp_port:="${UDP_PORT}" \
  -p topic:=/tracer/robot_odom_flat \
  > "$ROS2_LOG" 2>&1 &

PID=$!
echo "[TRACER] ROS2 odom receiver pid=$PID"

sleep 1.0

echo
echo "========== process check =========="
echo "[ROS1 model-state sender]"
docker exec a1_cpp_ctrl_docker bash -lc 'pgrep -af tracer_ros1_model_states_odom_udp_sender.py || true'

echo
echo "[ROS2 receiver]"
pgrep -af "tracer_udp_odom_to_ros2_node.py" || true

echo
echo "========== log tails =========="
echo "[ROS1 sender log]"
docker exec a1_cpp_ctrl_docker bash -lc "tail -n 20 ${ROS1_LOG} 2>/dev/null || true"

echo
echo "[ROS2 receiver log]"
tail -n 20 "$ROS2_LOG" 2>/dev/null || true
