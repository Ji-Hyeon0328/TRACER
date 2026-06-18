#!/usr/bin/env bash
set -eo pipefail

CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"

echo "[TRACER] restart A1-QP-MPC controller only"
echo "[TRACER] container: $CTRL_CONTAINER"

scripts/runtime/tracer_cleanup_stale_style_publishers.sh || true

sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
pkill -f "/root/A1_ctrl_ws/devel/lib/a1_cpp/gazebo_a1_ctrl" || true
pkill -f "roslaunch a1_cpp a1_ctrl.launch" || true
sleep 1
pgrep -af gazebo_a1_ctrl || true
' || true

scripts/runtime/tracer_start_a1_qpmc_controller_only.sh

echo
echo "[TRACER] verify /tracer/mpc_reference subscription"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rostopic info /tracer/mpc_reference || true
'
