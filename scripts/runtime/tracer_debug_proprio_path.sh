#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WS="$ROOT/ros2_ws"
PULSE_SEC="${TRACER_DEBUG_PULSE_SEC:-5.0}"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true

echo
echo "========== 1. ROS1 raw topic list =========="
sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
echo "[topics matching joint/imu/odom/contact/wrench]"
rostopic list | egrep "joint_states|trunk_imu|torso_odom|contact|wrench|foot|force" || true
' || true

echo
echo "========== 2. ROS1 proprio sender process/log =========="
sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
echo "[process]"
pgrep -af tracer_ros1_proprio_udp_sender.py || true
echo
echo "[log tail]"
tail -n 80 /tmp/tracer_ros1_proprio_udp_sender.log 2>/dev/null || true
' || true

echo
echo "========== 3. ROS2 proprio receiver process/log =========="
echo "[process]"
pgrep -af tracer_udp_proprio_to_ros2_node.py || true

echo
echo "[latest qwerty log dir]"
LATEST_QWERTY_LOG="$(ls -td "$ROOT"/logs/qwerty_* 2>/dev/null | head -1 || true)"
echo "$LATEST_QWERTY_LOG"

if [ -n "${LATEST_QWERTY_LOG:-}" ]; then
  echo
  echo "[ros2_proprio_receiver.log tail]"
  tail -n 80 "$LATEST_QWERTY_LOG/ros2_proprio_receiver.log" 2>/dev/null || true

  echo
  echo "[learned_encoder_udp_client.log tail]"
  tail -n 80 "$LATEST_QWERTY_LOG/learned_encoder_udp_client.log" 2>/dev/null || true

  echo
  echo "[proprio_encoder_udp_server_v0.log tail]"
  tail -n 80 "$LATEST_QWERTY_LOG/proprio_encoder_udp_server_v0.log" 2>/dev/null || true
fi

echo
echo "========== 4. Pulse physics and check raw ROS1 topics =========="
echo "[TRACER] unpause for ${PULSE_SEC}s"

sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/unpause_physics "{}"
' >/dev/null || true

sleep 1.0

echo
echo "[ROS1 /a1_gazebo/joint_states one msg]"
sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
timeout 3 rostopic echo /a1_gazebo/joint_states -n 1
' || true

echo
echo "[ROS1 /trunk_imu one msg]"
sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
timeout 3 rostopic echo /trunk_imu -n 1
' || true

echo
echo "[ROS1 /torso_odom one msg]"
sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
timeout 3 rostopic echo /torso_odom -n 1
' || true

echo
echo "[sleep remaining]"
sleep "$PULSE_SEC"

echo
echo "[TRACER] pause again"

sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/pause_physics "{}"
' >/dev/null || true

echo
echo "========== 5. ROS2 topic check after pulse =========="
echo
echo "[/tracer/proprio_vector]"
timeout 5 ros2 topic echo /tracer/proprio_vector --once || true

echo
echo "[/tracer/context_vector]"
timeout 5 ros2 topic echo /tracer/context_vector --once || true

echo
echo "[/tracer/latent_vector]"
timeout 5 ros2 topic echo /tracer/latent_vector --once || true

echo
echo "========== interpretation =========="
echo "If raw ROS1 joint/imu/odom topics are empty: Gazebo/robot sensor stream is not alive."
echo "If raw ROS1 topics exist but /tracer/proprio_vector is empty: check tracer_ros1_proprio_udp_sender.py or UDP 50130 receiver."
echo "If /tracer/proprio_vector exists but context/latent empty: check learned encoder client/server UDP 50200."
