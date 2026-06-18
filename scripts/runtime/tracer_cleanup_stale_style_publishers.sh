#!/usr/bin/env bash
set -eo pipefail

CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"

echo "[TRACER] cleanup stale style MPC reference publishers"

echo
echo "[TRACER] host cleanup..."
pkill -f "tracer_publish_style_mpc_ref_ros1.sh" || true
pkill -f "tracer_style_mpc_ref_ros1_publisher" || true

echo
echo "[TRACER] controller container cleanup..."
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
pkill -f "tracer_style_mpc_ref_ros1_publisher" || true
pkill -f "tracer_publish_style_mpc_ref_ros1.sh" || true
' || true

sleep 0.5

echo
echo "[TRACER] remaining ROS1 /tracer/mpc_reference topic info:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rostopic info /tracer/mpc_reference || true
'
