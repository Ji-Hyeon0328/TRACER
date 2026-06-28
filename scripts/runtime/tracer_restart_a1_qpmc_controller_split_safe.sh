#!/usr/bin/env bash
set -euo pipefail

CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"

echo "[TRACER] split-safe restart A1-QP-MPC controller"
echo "[TRACER] controller container: $CTRL_CONTAINER"

echo "[TRACER] build a1_cpp"
docker exec "$CTRL_CONTAINER" bash -lc '
  source /opt/ros/melodic/setup.bash
  source /root/unitree_ws/devel/setup.bash
  source /root/A1_ctrl_ws/devel/setup.bash
  cd /root/A1_ctrl_ws
  catkin build a1_cpp
'

echo "[TRACER] kill exact old controller processes"
docker exec "$CTRL_CONTAINER" bash -lc '
  set +e
  ps -eo pid,comm,args | awk '\''$2=="gazebo_a1_ctrl" || ($2=="python" && $0 ~ /\/opt\/ros\/melodic\/bin\/roslaunch/ && $0 ~ /a1_ctrl.launch/) {print $1}'\'' \
    | xargs -r kill -9
  sleep 1
'

echo "[TRACER] start controller"
docker exec -d "$CTRL_CONTAINER" bash -lc '
  source /opt/ros/melodic/setup.bash
  source /root/unitree_ws/devel/setup.bash
  source /root/A1_ctrl_ws/devel/setup.bash
  roslaunch a1_cpp a1_ctrl.launch type:=gazebo solver_type:=mpc \
    > /tmp/tracer_a1_ctrl.launch.log 2>&1
'

sleep 6

echo "[TRACER] verify controller and /tracer/mpc_reference"
docker exec "$CTRL_CONTAINER" bash -lc '
  source /opt/ros/melodic/setup.bash
  source /root/unitree_ws/devel/setup.bash
  source /root/A1_ctrl_ws/devel/setup.bash

  echo "=== process ==="
  ps -eo pid,comm,args | awk '\''$2=="gazebo_a1_ctrl" || ($2=="python" && $0 ~ /\/opt\/ros\/melodic\/bin\/roslaunch/ && $0 ~ /a1_ctrl.launch/) {print}'\'' || true

  echo "=== rosnode list ==="
  rosnode list | grep -Ei "gazebo_a1_ctrl|a1|tracer" || true

  echo "=== /tracer/mpc_reference ==="
  rostopic info /tracer/mpc_reference || true

  echo "=== controller log tail ==="
  tail -80 /tmp/tracer_a1_ctrl.launch.log || true
'
