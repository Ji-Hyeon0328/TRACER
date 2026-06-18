#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WS="$ROOT/ros2_ws"
PULSE_SEC="${TRACER_QWERTY_PULSE_SEC:-5.0}"

LOG_DIR="$ROOT/logs/qwerty_live_precheck_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true

echo
echo "========== qwerty process sanity =========="

echo
echo "[ROS1 container processes]"
sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
echo "[proprio sender]"
pgrep -af tracer_ros1_proprio_udp_sender.py || true
echo
echo "[mpc ref receiver]"
pgrep -af tracer_udp_to_ros1_mpc_ref.py || true
' || true

echo
echo "[host qwerty processes]"
pgrep -af tracer_udp_proprio_to_ros2_node.py || true
pgrep -af tracer_ros2_mpc_ref_udp_sender.py || true
pgrep -af tracer_learned_encoder_udp_client_node.py || true
pgrep -af tracer_objective_selector_stub_node.py || true
pgrep -af tracer_learned_high_level_policy_udp_client_v1_node.py || true
pgrep -af tracer_proprio_encoder_udp_server_v0.py || true
pgrep -af tracer_high_level_policy_udp_server_v1.py || true

echo
echo "========== qwerty node graph =========="
ros2 node list | grep tracer || true

echo
echo "========== start topic probes BEFORE physics pulse =========="
echo "[TRACER] logs: $LOG_DIR"

timeout 8 ros2 topic echo /tracer/proprio_vector --once > "$LOG_DIR/proprio_vector.txt" 2>&1 &
PID_PROP=$!

timeout 8 ros2 topic echo /tracer/context_vector --once > "$LOG_DIR/context_vector.txt" 2>&1 &
PID_CTX=$!

timeout 8 ros2 topic echo /tracer/latent_vector --once > "$LOG_DIR/latent_vector.txt" 2>&1 &
PID_LAT=$!

timeout 8 ros2 topic echo /tracer/objective_weights --once > "$LOG_DIR/objective_weights.txt" 2>&1 &
PID_OBJ=$!

timeout 8 ros2 topic echo /tracer/mpc_reference --once > "$LOG_DIR/mpc_reference.txt" 2>&1 &
PID_MPC=$!

sleep 0.5

echo
echo "========== pulse Gazebo physics while probes are active =========="
echo "[TRACER] unpause for ${PULSE_SEC}s"

sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/unpause_physics "{}"
' >/dev/null || true

sleep "$PULSE_SEC"

echo "[TRACER] pause again"

sudo docker exec a1_cpp_ctrl_docker bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/pause_physics "{}"
' >/dev/null || true

wait "$PID_PROP" || true
wait "$PID_CTX" || true
wait "$PID_LAT" || true
wait "$PID_OBJ" || true
wait "$PID_MPC" || true

echo
echo "========== qwerty live topic check =========="

show_file () {
  local label="$1"
  local file="$2"
  echo
  echo "[$label]"
  if [ -s "$file" ]; then
    cat "$file"
  else
    echo "<empty>"
  fi
}

show_file "/tracer/proprio_vector" "$LOG_DIR/proprio_vector.txt"
show_file "/tracer/context_vector" "$LOG_DIR/context_vector.txt"
show_file "/tracer/latent_vector" "$LOG_DIR/latent_vector.txt"
show_file "/tracer/objective_weights" "$LOG_DIR/objective_weights.txt"
show_file "/tracer/mpc_reference" "$LOG_DIR/mpc_reference.txt"

echo
echo "========== latest receiver/client logs =========="
LATEST_QWERTY_LOG="$(ls -td "$ROOT"/logs/qwerty_[0-9]* 2>/dev/null | head -1 || true)"
echo "[latest qwerty log dir] $LATEST_QWERTY_LOG"

if [ -n "${LATEST_QWERTY_LOG:-}" ]; then
  echo
  echo "[ros2_proprio_receiver.log tail]"
  tail -n 20 "$LATEST_QWERTY_LOG/ros2_proprio_receiver.log" 2>/dev/null || true

  echo
  echo "[learned_encoder_udp_client.log tail]"
  tail -n 20 "$LATEST_QWERTY_LOG/learned_encoder_udp_client.log" 2>/dev/null || true
fi

echo
echo "========== expected =========="
echo "proprio_vector:      should publish during physics pulse"
echo "context_vector:      should publish after encoder response"
echo "latent_vector:       should publish after encoder response"
echo "objective_weights:   should publish"
echo "mpc_reference:       stop command is OK in qwerty-only state"
echo
