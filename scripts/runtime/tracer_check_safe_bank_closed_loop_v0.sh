#!/usr/bin/env bash
set -euo pipefail

echo "================ ROS2 policy / selected theta ================"
set +u
source /opt/ros/humble/setup.bash
if [ -f ros2_ws/install/setup.bash ]; then
  source ros2_ws/install/setup.bash
fi
set -u

pgrep -af 'tracer_safe_bank_policy_node_v0.py' || true
pgrep -af 'tracer_ros2_mpc_ref_udp_sender.py' || true

echo
echo "=== ROS2 /tracer/selected_theta ==="
timeout 3 ros2 topic echo --once /tracer/selected_theta || true

echo
echo "=== ROS2 /tracer/mpc_reference ==="
timeout 3 ros2 topic echo --once /tracer/mpc_reference || true

echo
echo "=== ROS2 topic info ==="
ros2 topic info -v /tracer/mpc_reference || true

echo
echo "================ ROS1 bridge/controller ================"
docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash 2>/dev/null || true
source /root/A1_ctrl_ws/devel/setup.bash 2>/dev/null || true

echo "=== ROS1 bridge process ==="
pgrep -af tracer_udp_to_ros1_mpc_ref.py || true

echo
echo "=== ROS1 /tracer/mpc_reference info ==="
rostopic info /tracer/mpc_reference || true

echo
echo "=== ROS1 /tracer/mpc_reference echo ==="
timeout 3 rostopic echo -n 1 /tracer/mpc_reference || true
'
