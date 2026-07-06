#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
CTRL_CONTAINER="${TRACER_CTRL_CONTAINER:-a1_cpp_ctrl_docker}"
UDP_PORT="${TRACER_STATE_UDP_PORT:-50130}"

ROS1_SENDER_HOST="$ROOT/scripts/runtime/tracer_ros1_state_udp_sender.py"
ROS1_SENDER_DOCKER="/tmp/tracer_ros1_state_udp_sender.py"

echo "[TRACER] starting ROS1->ROS2 state bridge on UDP port ${UDP_PORT}"

if [[ ! -f "$ROS1_SENDER_HOST" ]]; then
  echo "[TRACER][ERROR] missing sender script: $ROS1_SENDER_HOST"
  exit 1
fi

pkill -f "tracer_udp_state_to_ros2_node.py" 2>/dev/null || true

docker exec "$CTRL_CONTAINER" bash -lc '
pkill -f "tracer_ros1_state_udp_sender.py" 2>/dev/null || true
' || true

echo "[TRACER] copying ROS1 sender into docker: $ROS1_SENDER_DOCKER"
docker cp "$ROS1_SENDER_HOST" "$CTRL_CONTAINER:$ROS1_SENDER_DOCKER"

docker exec -d "$CTRL_CONTAINER" bash --noprofile --norc -lc "
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash 2>/dev/null || true
source /root/A1_ctrl_ws/devel/setup.bash 2>/dev/null || true
chmod +x ${ROS1_SENDER_DOCKER}
python ${ROS1_SENDER_DOCKER} \
  _udp_ip:=127.0.0.1 \
  _udp_port:=${UDP_PORT} \
  _pub_hz:=50.0 \
  > /tmp/tracer_ros1_state_udp_sender.log 2>&1
"

set +u
source /opt/ros/humble/setup.bash

nohup /usr/bin/python3 "$ROOT/scripts/runtime/tracer_udp_state_to_ros2_node.py" \
  --ros-args \
  -p bind_ip:=0.0.0.0 \
  -p bind_port:=${UDP_PORT} \
  -p publish_hz:=50.0 \
  > /tmp/tracer_udp_state_to_ros2_node.log 2>&1 &

echo $! > /tmp/tracer_udp_state_to_ros2_node.pid

sleep 1.0

echo "[TRACER] ROS2 state topic:"
ros2 topic list -t | grep -E "/tracer/robot_state" || true

echo
echo "[TRACER] sample:"
timeout 3s ros2 topic echo --once /tracer/robot_state_flat || true
