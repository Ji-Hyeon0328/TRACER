#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"
PORT="${TRACER_MPC_REF_UDP_PORT:-50110}"

echo "[TRACER] direct UDP -> ROS1 /tracer/mpc_reference precheck"

scripts/runtime/tracer_ensure_mpc_ref_bridge.sh

echo
echo "[TRACER] start ROS1 echo..."
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
timeout 8 rostopic echo /tracer/mpc_reference -n 1
' > /tmp/tracer_ros1_mpc_ref_echo_direct.txt 2>&1 &
ECHO_PID=$!

sleep 1.0

echo
echo "[TRACER] send direct UDP test command..."
python3 - <<PY
import socket, struct, time
port = int("${PORT}")
data = [9999.0, 0.22, 0.0, 0.30, 0.035, 1.0]
pkt = struct.pack("<6d", *data)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for _ in range(10):
    sock.sendto(pkt, ("127.0.0.1", port))
    time.sleep(0.05)
print("sent", data, "to UDP", port)
PY

wait "$ECHO_PID" || true

echo
echo "========== ROS1 echo result =========="
cat /tmp/tracer_ros1_mpc_ref_echo_direct.txt || true

echo
echo "========== topic info =========="
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rostopic info /tracer/mpc_reference || true
'
