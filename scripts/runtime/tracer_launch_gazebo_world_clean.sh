#!/usr/bin/env bash
set -euo pipefail

WORLD_NAME="${1:-earth}"
GAZEBO_CONTAINER="${TRACER_GAZEBO_CONTAINER:-a1_unitree_gazebo_docker}"
CTRL_CONTAINER="${TRACER_CTRL_CONTAINER:-a1_cpp_ctrl_docker}"
RNAME="${TRACER_ROBOT_NAME:-a1}"

echo "[TRACER] clean launch Gazebo world"
echo "[TRACER] world: $WORLD_NAME"
echo "[TRACER] gazebo container: $GAZEBO_CONTAINER"
echo "[TRACER] ctrl container:    $CTRL_CONTAINER"

echo
echo "========== stop host-side lite helpers =========="
cd "$(dirname "$0")/../.."
scripts/runtime/tracer_stop_data_collection_lite.sh || true

echo
echo "========== restart containers =========="
docker restart "$GAZEBO_CONTAINER"
docker restart "$CTRL_CONTAINER"
sleep 5

echo
echo "========== launch gazebo world =========="
docker exec -d "$GAZEBO_CONTAINER" bash --noprofile --norc -lc "
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
roslaunch unitree_gazebo normal.launch rname:=${RNAME} wname:=${WORLD_NAME} \
  > /tmp/tracer_gazebo_${WORLD_NAME}.launch.log 2>&1
"

sleep 6

echo
echo "========== verify gazebo world =========="
docker exec "$GAZEBO_CONTAINER" bash --noprofile --norc -lc "
pgrep -af 'gzserver|normal.launch|${WORLD_NAME}|sponge|slippery|earth' || true

set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash 2>/dev/null || true

echo
echo '===== /gazebo/get_world_properties ====='
rosservice call /gazebo/get_world_properties '{}' || true
"

echo
echo "========== launch controller =========="
docker exec -d "$CTRL_CONTAINER" bash --noprofile --norc -lc "
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
roslaunch a1_cpp a1_ctrl.launch type:=gazebo solver_type:=mpc \
  > /tmp/tracer_a1_ctrl.launch.log 2>&1
"

sleep 4

echo
echo "========== verify controller =========="
docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc "
pgrep -af 'a1_ctrl|gazebo_a1_ctrl|mpc|wbc' || true

set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash

echo
echo '===== /tracer/mpc_reference topic info ====='
rostopic info /tracer/mpc_reference || true
"

echo
echo "[TRACER] clean world launch done: $WORLD_NAME"
