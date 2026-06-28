#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WS="$ROOT/ros2_ws"
LOG_DIR="$ROOT/logs/qwerty_$(date +%Y%m%d_%H%M%S)"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"

mkdir -p "$LOG_DIR"

echo "[TRACER] start qwerty state"
echo "[TRACER] root: $ROOT"
echo "[TRACER] logs: $LOG_DIR"

if [ ! -d "$WS" ]; then
  echo "[TRACER][ERROR] ROS2 workspace not found: $WS"
  exit 1
fi

if [ ! -f "$CONDA_SH" ]; then
  echo "[TRACER][ERROR] conda.sh not found: $CONDA_SH"
  echo "Set CONDA_SH=/path/to/conda.sh if needed."
  exit 1
fi

echo
echo "[TRACER] cleaning old qwerty host processes..."
pkill -9 -f tracer_udp_proprio_to_ros2_node.py || true
pkill -9 -f tracer_ros2_mpc_ref_udp_sender.py || true
pkill -9 -f tracer_learned_encoder_udp_client_node.py || true
pkill -9 -f tracer_objective_selector_stub_node.py || true
pkill -9 -f tracer_learned_high_level_policy_udp_client_v1_node.py || true
pkill -9 -f tracer_proprio_encoder_udp_server_v0.py || true
pkill -9 -f tracer_high_level_policy_udp_server_v1.py || true

echo "[TRACER] cleaning old qwerty ROS1 container processes..."
sudo docker exec a1_cpp_ctrl_docker bash -lc '
pkill -9 -f tracer_ros1_proprio_udp_sender.py || true
pkill -9 -f tracer_udp_to_ros1_mpc_ref.py || true
' || true

echo
echo "[TRACER] starting ROS1 container nodes..."

sudo docker exec -d a1_cpp_ctrl_docker bash -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rosrun a1_cpp tracer_ros1_proprio_udp_sender.py _odom_topic:=/body_pose_ground_truth > /tmp/tracer_ros1_proprio_udp_sender.log 2>&1
'

sudo docker exec -d a1_cpp_ctrl_docker bash -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rosrun a1_cpp tracer_udp_to_ros1_mpc_ref.py > /tmp/tracer_udp_to_ros1_mpc_ref.log 2>&1
'

start_ros2_node () {
  local name="$1"
  local script="$2"
  shift 2

  echo "[TRACER] starting ROS2 node: $name"
  nohup bash -lc "
    cd '$WS'
    source /opt/ros/humble/setup.bash
    source install/setup.bash 2>/dev/null || true
    exec /usr/bin/python3 '$WS/src/tracer_a1_qpmc_adapter/scripts/$script' $*
  " > "$LOG_DIR/${name}.log" 2>&1 &

  echo $! > "$LOG_DIR/${name}.pid"
  sleep 0.3
}

start_conda_node () {
  local name="$1"
  shift

  echo "[TRACER] starting env_isaaclab node: $name"
  nohup bash -lc "
    source '$CONDA_SH'
    conda activate env_isaaclab
    cd '$ROOT'
    exec $*
  " > "$LOG_DIR/${name}.log" 2>&1 &

  echo $! > "$LOG_DIR/${name}.pid"
  sleep 0.5
}

echo
echo "[TRACER] starting env_isaaclab UDP servers..."

start_conda_node "proprio_encoder_udp_server_v0" \
  python3 learning/tracer_proprio_encoder_udp_server_v0.py \
  --checkpoint "$ROOT/checkpoints/tracer_proprio_encoder_v0_best.pt"

start_conda_node "high_level_policy_udp_server_v1" \
  python3 learning/tracer_high_level_policy_udp_server_v1.py \
  --checkpoint "$ROOT/checkpoints/tracer_high_level_policy_v1_best.pt"

echo
echo "[TRACER] starting ROS2 host qwerty nodes..."

start_ros2_node "ros2_proprio_receiver" \
  tracer_udp_proprio_to_ros2_node.py

start_ros2_node "ros2_mpc_ref_sender" \
  tracer_ros2_mpc_ref_udp_sender.py

start_ros2_node "learned_encoder_udp_client" \
  tracer_learned_encoder_udp_client_node.py

start_ros2_node "objective_selector_stub" \
  tracer_objective_selector_stub_node.py

start_ros2_node "learned_high_level_policy_udp_client_v1" \
  tracer_learned_high_level_policy_udp_client_v1_node.py

echo
echo "[TRACER] qwerty state launch commands issued."
echo "[TRACER] This script intentionally does NOT start odom/waypoint overlay."
echo "[TRACER] For waypoint mission, run:"
echo "  scripts/runtime/tracer_start_waypoint_overlay_from_qwerty.sh"
echo
echo "[TRACER] Check status with:"
echo "  scripts/runtime/tracer_precheck_qwerty_state.sh"
echo
echo "[TRACER] Logs:"
echo "  $LOG_DIR"
