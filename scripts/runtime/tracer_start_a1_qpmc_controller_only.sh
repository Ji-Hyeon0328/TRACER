#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"

echo "[TRACER] start A1-QP-MPC controller only"
echo "[TRACER] container: $CTRL_CONTAINER"

echo
echo "[TRACER] current controller-related processes before start:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
ps -eo pid,comm,args | awk '\''$2=="gazebo_a1_ctrl" || ($2=="python" && $0 ~ /roslaunch/ && $0 ~ /a1_ctrl.launch/) {print}'\'' || true
'

echo
echo "[TRACER] start roslaunch a1_cpp a1_ctrl.launch type:=gazebo solver_type:=mpc"
sudo docker exec -d "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
roslaunch a1_cpp a1_ctrl.launch type:=gazebo solver_type:=mpc > /tmp/tracer_a1_ctrl.launch.log 2>&1
'

echo
echo "[TRACER] waiting for controller startup..."
sleep 8

echo
echo "[TRACER] controller-related processes after start:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
ps -eo pid,comm,args | awk '\''$2=="gazebo_a1_ctrl" || ($2=="python" && $0 ~ /roslaunch/ && $0 ~ /a1_ctrl.launch/) {print}'\'' || true
'

echo
echo "[TRACER] rosnode list:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rosnode list | grep -E "gazebo_a1_ctrl|a1" || true
'

echo
echo "[TRACER] launch log tail:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
tail -n 160 /tmp/tracer_a1_ctrl.launch.log 2>/dev/null || true
'

echo
echo "[TRACER] /tracer/mpc_reference topic info:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rostopic info /tracer/mpc_reference || true
'
