#!/usr/bin/env bash
set -eo pipefail

CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"

echo "[TRACER] cleanup stale style MPC reference publishers"

echo
echo "[TRACER] host process cleanup..."
pkill -f "tracer_publish_style_mpc_ref_ros1.sh" || true
pkill -f "tracer_style_mpc_ref_ros1_publisher" || true
pkill -f "rostopic pub.*/tracer/mpc_reference" || true

echo
echo "[TRACER] controller container ROS node/process cleanup..."
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash

echo "[TRACER] ROS nodes before cleanup:"
rosnode list 2>/dev/null | grep -E "tracer_style_mpc_ref|style_mpc|mpc_ref_ros1_publisher" || true

for n in $(rosnode list 2>/dev/null | grep -E "tracer_style_mpc_ref|style_mpc|mpc_ref_ros1_publisher" || true); do
  echo "[TRACER] rosnode kill $n"
  rosnode kill "$n" || true
done

pkill -f "tracer_publish_style_mpc_ref_ros1.sh" || true
pkill -f "tracer_style_mpc_ref_ros1_publisher" || true
pkill -f "rostopic pub.*/tracer/mpc_reference" || true
sleep 0.5

echo "[TRACER] ROS nodes after cleanup:"
rosnode list 2>/dev/null | grep -E "tracer_style_mpc_ref|style_mpc|mpc_ref_ros1_publisher" || true
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

echo
echo "[TRACER] assert no stale style publisher remains"
if sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rostopic info /tracer/mpc_reference 2>/dev/null | grep -q "tracer_style_mpc_ref"
'; then
  echo "[ERROR] stale style publisher is still alive on /tracer/mpc_reference"
  exit 40
fi

echo "[TRACER] stale style publisher cleanup passed"
