#!/usr/bin/env bash
set -eo pipefail

CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"

echo "[TRACER] kill stale rostopic publishers on /tracer/mpc_reference"

sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash

echo "========== current /tracer/mpc_reference =========="
rostopic info /tracer/mpc_reference || true

echo
echo "========== rostopic nodes =========="
rosnode list | grep rostopic || true

for n in $(rosnode list | grep rostopic || true); do
  uri=$(rosnode info "$n" 2>/dev/null | awk "/Pid:/ {print \$2; exit}")
  echo "[TRACER] rosnode kill $n"
  rosnode kill "$n" || true
done
'

echo
echo "[TRACER] trying host-side port cleanup for stale rostopic XMLRPC nodes..."
for port in $(sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rostopic info /tracer/mpc_reference 2>/dev/null | grep rostopic_ | sed -E "s/.*:([0-9]+).*/\1/" || true
'); do
  echo "[TRACER] fuser kill tcp/$port"
  sudo fuser -k "${port}/tcp" || true
done

sleep 1

echo
echo "[TRACER] after cleanup:"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rostopic info /tracer/mpc_reference || true
'
