#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

GAZEBO_CONTAINER="${TRACER_GAZEBO_CONTAINER:-a1_unitree_gazebo_docker}"
CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"
MODEL_NAME="${TRACER_A1_MODEL_NAME:-a1_gazebo}"

echo
echo "========== reset/qwer state ROS1 process check =========="

echo
echo "[controller container: tracer_udp_to_ros1_mpc_ref]"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
ps aux | grep "[t]racer_udp_to_ros1_mpc_ref.py" || true
'

echo
echo "[gazebo container: reset helpers should NOT still be running]"
sudo docker exec "$GAZEBO_CONTAINER" bash --noprofile --norc -lc '
ps aux | grep "[u]nitree_servo" || true
ps aux | grep "[u]nitree_move_kinetic" || true
'

echo
echo "========== Gazebo physics state =========="
sudo docker exec "$GAZEBO_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
timeout 3 rosservice call /gazebo/get_physics_properties "{}" | grep -E "pause|time_step|max_update_rate" || true
'

echo
echo "========== Gazebo model pose snapshot via service =========="
sudo docker exec "$GAZEBO_CONTAINER" bash --noprofile --norc -lc "
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
echo '[TRACER] model_name=${MODEL_NAME}'
timeout 3 rosservice call /gazebo/get_model_state \"model_name: '${MODEL_NAME}'
relative_entity_name: 'world'\" | sed -n '/pose:/,/twist:/p' || true
"

echo
echo "========== expected =========="
echo "tracer_udp_to_ros1_mpc_ref.py: running in controller container"
echo "unitree_servo / unitree_move_kinetic: not running in gazebo container"
echo "Gazebo physics pause: True"
echo "robot pose: standing near zero point"
