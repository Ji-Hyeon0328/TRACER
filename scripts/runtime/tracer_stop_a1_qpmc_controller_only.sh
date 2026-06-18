#!/usr/bin/env bash
set -eo pipefail

CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"

echo "[TRACER] stop A1-QP-MPC controller only"
echo "[TRACER] container: $CTRL_CONTAINER"

sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash

for n in $(rosnode list 2>/dev/null | grep -E "^/gazebo_a1_ctrl$" || true); do
  echo "[TRACER] rosnode kill $n"
  rosnode kill "$n" || true
done

pkill -f "/root/A1_ctrl_ws/devel/lib/a1_cpp/gazebo_a1_ctrl" || true
pkill -f "roslaunch a1_cpp a1_ctrl.launch" || true
sleep 0.5

echo "[TRACER] remaining controller processes:"
pgrep -af gazebo_a1_ctrl || true
'
