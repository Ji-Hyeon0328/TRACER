#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"

echo "[TRACER] ensure ROS1 MPC reference bridge"
echo "[TRACER] controller container: $CTRL_CONTAINER"

echo
echo "[TRACER] killing old tracer_udp_to_ros1_mpc_ref.py if any..."
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
pkill -9 -f tracer_udp_to_ros1_mpc_ref.py || true
' || true

sleep 0.3

echo
echo "[TRACER] starting tracer_udp_to_ros1_mpc_ref.py..."
sudo docker exec -d "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rosrun a1_cpp tracer_udp_to_ros1_mpc_ref.py > /tmp/tracer_udp_to_ros1_mpc_ref.log 2>&1
'

sleep 1.0

echo
echo "[TRACER] process check:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
ps aux | grep "[t]racer_udp_to_ros1_mpc_ref.py" || true
'

echo
echo "[TRACER] ROS1 /tracer/mpc_reference topic info:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
timeout 3 rostopic info /tracer/mpc_reference || true
'

echo
echo "[TRACER] receiver log tail:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
tail -n 40 /tmp/tracer_udp_to_ros1_mpc_ref.log 2>/dev/null || true
'
