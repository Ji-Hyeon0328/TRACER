#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WS="$ROOT/ros2_ws"
CTRL_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}"
LOG_DIR="${ROOT}/logs/data_collection_lite_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

echo "[TRACER] start data_collection_lite"
echo "[TRACER] root: $ROOT"
echo "[TRACER] logs: $LOG_DIR"
echo "[TRACER] controller container: $CTRL_CONTAINER"

if [ ! -d "$WS" ]; then
  echo "[TRACER][ERROR] ROS2 workspace not found: $WS"
  exit 1
fi

echo
echo "========== kill conflicting high-level publishers ==========\necho "[TRACER] kill Phase-A runtime policy publishers"
pkill -9 -f tracer_phase_a_ram_aware_policy_node_v0.py 2>/dev/null || true
pkill -9 -f tracer_phase_a_override_policy_node_v0.py 2>/dev/null || true
"
pkill -9 -f tracer_fusion_policy_mpc_ref_node.py 2>/dev/null || true
pkill -9 -f tracer_objective_selector_stub_node.py 2>/dev/null || true
pkill -9 -f tracer_learned_high_level_policy_udp_client_v1_node.py 2>/dev/null || true
pkill -9 -f tracer_learned_high_level_policy_udp_client_node.py 2>/dev/null || true
pkill -9 -f tracer_high_level_controller_stub_node.py 2>/dev/null || true
pkill -9 -f tracer_runtime_validated_policy_node_v0.py 2>/dev/null || true
pkill -9 -f tracer_phase_a_ram_aware_policy_node_v0.py 2>/dev/null || true
pkill -9 -f tracer_safe_bank_policy_node_v0.py 2>/dev/null || true
pkill -9 -f tracer_objective_selector_beta_node_v0.py 2>/dev/null || true
pkill -9 -f "ros2 topic pub.*/tracer/mpc_reference" 2>/dev/null || true

echo
echo "========== restart ROS1 proprio UDP sender =========="
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
pkill -9 -f tracer_ros1_proprio_udp_sender.py || true
' || true

sudo docker exec -d "$CTRL_CONTAINER" bash --noprofile --norc -lc '
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rosrun a1_cpp tracer_ros1_proprio_udp_sender.py _odom_topic:=/body_pose_ground_truth > /tmp/tracer_ros1_proprio_udp_sender.log 2>&1
'

echo
echo "========== restart ROS2 proprio UDP receiver =========="
pkill -9 -f tracer_udp_proprio_to_ros2_node.py 2>/dev/null || true

nohup bash -lc "
  set +u
  source /opt/ros/humble/setup.bash
  if [ -f '$WS/install/setup.bash' ]; then
    source '$WS/install/setup.bash'
  fi
  exec /usr/bin/python3 '$WS/src/tracer_a1_qpmc_adapter/scripts/tracer_udp_proprio_to_ros2_node.py'
" > "$LOG_DIR/ros2_proprio_receiver.log" 2>&1 &

echo $! > "$LOG_DIR/ros2_proprio_receiver.pid"

echo
echo "========== ensure ROS1 MPC ref bridge =========="
"$ROOT/scripts/runtime/tracer_ensure_mpc_ref_bridge.sh"

echo
echo "========== ensure ROS2 MPC ref UDP sender =========="
"$ROOT/scripts/runtime/tracer_ensure_ros2_mpc_ref_udp_sender.sh"

echo
echo "========== start odom bridge only =========="
"$ROOT/scripts/runtime/tracer_start_odom_bridge_only.sh"

echo
echo "========== process check =========="
echo "[host]"
pgrep -af "tracer_udp_proprio_to_ros2_node.py|tracer_udp_odom_to_ros2_node.py|tracer_ros2_mpc_ref_udp_sender.py|tracer_fusion_policy_mpc_ref_node.py|tracer_objective_selector_stub_node.py|tracer_learned_high_level_policy_udp_client" || true

echo
echo "[controller container]"
sudo docker exec "$CTRL_CONTAINER" bash --noprofile --norc -lc '
pgrep -af "tracer_ros1_proprio_udp_sender.py|tracer_ros1_odom_udp_sender.py|tracer_udp_to_ros1_mpc_ref.py" || true
'

echo
echo "========== ROS2 /tracer/mpc_reference topic info =========="
set +u
source /opt/ros/humble/setup.bash
if [ -f "$WS/install/setup.bash" ]; then
  source "$WS/install/setup.bash"
fi
set -u
ros2 topic info -v /tracer/mpc_reference || true

echo
echo "[TRACER] data_collection_lite started."
echo "[TRACER] Expected for reference sweep:"
echo "  - /tracer/mpc_reference subscriber: tracer_ros2_mpc_ref_udp_sender"
echo "  - no learned/fusion/objective publisher"
echo "  - reference sweep runner will be the only publisher"
