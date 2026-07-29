#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"

export TRACER_CTRL_CONTAINER="${TRACER_CTRL_CONTAINER:-a1_cpp_ctrl_ll1_docker}"
export TRACER_A1_CONTAINER="${TRACER_A1_CONTAINER:-a1_cpp_ctrl_ll1_docker}"
WORLD="${TRACER_PHASE_C_WORLD:-${TRACER_PHASE_D4_WORLD:-tracer_mixed_solid_course_v0}}"
GOAL_X="${TRACER_PHASE_D4_GOAL_X:-${TRACER_PHASE_C_GOAL_X:-8.0}}"
PUB_HZ="${TRACER_PHASE_D4_PUB_HZ:-${TRACER_PHASE_C_PUB_HZ:-10}}"
VX="${TRACER_PHASE_C_FIXED_VX:-0.16}"
BODY_H="${TRACER_PHASE_C_FIXED_BODY_H:-0.32}"
CLEARANCE="${TRACER_PHASE_C_FIXED_CLEARANCE:-0.045}"

export TRACER_PHASE_D4_GOAL_X="$GOAL_X"
export TRACER_PHASE_D4_PUB_HZ="$PUB_HZ"
LOG_DIR="$ROOT/logs/phase_d4_context_meta_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "[TRACER] start Phase-D4 context-meta stack v0"
echo "[TRACER] world:     $WORLD"
echo "[TRACER] goal_x:    $GOAL_X"
echo "[TRACER] vx:        $VX"
echo "[TRACER] body_h:    $BODY_H"
echo "[TRACER] clearance: $CLEARANCE"
echo "[TRACER] logs:      $LOG_DIR"

set +u
source /opt/ros/humble/setup.bash
set -u

echo
echo "========== 0. stop stale Phase-C processes =========="
scripts/runtime/tracer_stop_phase_c_mixed_stack_v0.sh || true

echo
echo "========== 1. launch Gazebo world + frozen A1-QP-MPC =========="
/tmp/tracer_launch_gazebo_world_ll1_residual_v2.sh "$WORLD"


echo
echo "========== 2. qwer standing reset =========="
scripts/runtime/tracer_reset_to_qwer_state.sh

if [ -n "${TRACER_RESET_Y_OFFSET:-}" ]; then
  echo "[TRACER] applying requested reset y-offset after qwer reset: ${TRACER_RESET_Y_OFFSET}"
  scripts/runtime/tracer_apply_reset_y_offset_v0.sh
fi

echo
echo "========== 3. validate standing pose =========="
timeout -s INT -k 2s 20s /tmp/tracer_validate_a1_ready_standing_pose_nosudo.sh || echo '[WARN] D4 ready-standing validation timed out; continuing smoke test'

echo
echo "

========== 4. start bridges =========="
scripts/runtime/tracer_ensure_ros2_mpc_ref_udp_sender.sh
scripts/runtime/tracer_ensure_mpc_ref_bridge.sh
/tmp/tracer_start_modelstate_odom_bridge_ll1_v2.sh

echo
echo "========== 5. start mixed terrain context provider =========="
nohup /usr/bin/python3 \
  scripts/runtime/tracer_mixed_terrain_context_provider_v0.py \
  > "$LOG_DIR/context_provider.log" 2>&1 &
echo $! > "$LOG_DIR/context_provider.pid"
sleep 1.0

echo
echo "========== 6. start D4 context-meta selector node =========="
nohup /usr/bin/python3 \
  /tmp/tracer_phase_d4_context_meta_selector_node_v0.py \
  > "$LOG_DIR/context_meta_selector.log" 2>&1 &
echo $! > "$LOG_DIR/segment_action.pid"
sleep 1.0

echo
echo "========== 6b. start LL1 CSV recorders =========="

docker exec \
  "$TRACER_A1_CONTAINER" \
  bash --noprofile --norc -lc '
    set +e

    pkill -INT -f \
      "[r]ostopic echo -p /tracer/mpc_reference" \
      || true

    pkill -INT -f \
      "[r]ostopic echo -p /tracer/lowlevel/swing_apex_debug" \
      || true
  ' \
  || true

sleep 0.3

nohup docker exec \
  "$TRACER_A1_CONTAINER" \
  bash --noprofile --norc -lc '
    set +u
    source /opt/ros/melodic/setup.bash
    source /root/unitree_ws/devel/setup.bash
    source /root/A1_ctrl_ws/devel/setup.bash

    exec rostopic echo \
      -p \
      /tracer/mpc_reference
  ' \
  > "$LOG_DIR/ros1_mpc_reference.csv" \
  2> "$LOG_DIR/ros1_mpc_reference_recorder.log" &

echo $! > "$LOG_DIR/ros1_mpc_reference_recorder.pid"

nohup docker exec \
  "$TRACER_A1_CONTAINER" \
  bash --noprofile --norc -lc '
    set +u
    source /opt/ros/melodic/setup.bash
    source /root/unitree_ws/devel/setup.bash
    source /root/A1_ctrl_ws/devel/setup.bash

    exec rostopic echo \
      -p \
      /tracer/lowlevel/swing_apex_debug
  ' \
  > "$LOG_DIR/ros1_swing_apex_debug.csv" \
  2> "$LOG_DIR/ros1_swing_apex_debug_recorder.log" &

echo $! > "$LOG_DIR/ros1_swing_apex_debug_recorder.pid"

sleep 1.0

echo
echo "========== 6c. start joint energy proxy logger =========="

ENERGY_LOGGER_HOST="${TRACER_ENERGY_LOGGER_HOST:-/tmp/tracer_ros1_joint_energy_proxy_logger_v0.py}"
ENERGY_LOGGER_CONTAINER="/tmp/tracer_ros1_joint_energy_proxy_logger_v0.py"

test -s "$ENERGY_LOGGER_HOST" || {
  echo "[ERROR] missing energy logger: $ENERGY_LOGGER_HOST"
  exit 1
}

docker exec \
  "$TRACER_A1_CONTAINER" \
  bash --noprofile --norc -lc '
    set +e

    pkill -INT -f \
      "[t]racer_ros1_joint_energy_proxy_logger_v0.py" \
      || true
  ' \
  || true

docker cp \
  "$ENERGY_LOGGER_HOST" \
  "$TRACER_A1_CONTAINER:$ENERGY_LOGGER_CONTAINER"

nohup docker exec \
  "$TRACER_A1_CONTAINER" \
  bash --noprofile --norc -lc '
    set +u

    source /opt/ros/melodic/setup.bash
    source /root/unitree_ws/devel/setup.bash 2>/dev/null || true
    source /root/A1_ctrl_ws/devel/setup.bash 2>/dev/null || true

    exec python -u \
      /tmp/tracer_ros1_joint_energy_proxy_logger_v0.py \
      _publish_hz:=50.0 \
      _max_age_s:=0.20 \
      _ref_max_age_s:=0.75 \
      _model_name:=a1_gazebo
  ' \
  > "$LOG_DIR/ros1_joint_energy_proxy.csv" \
  2> "$LOG_DIR/ros1_joint_energy_proxy_recorder.log" &

echo $! > "$LOG_DIR/ros1_joint_energy_proxy_recorder.pid"

sleep 1.0

echo "[TRACER] energy logger process:"

docker exec \
  "$TRACER_A1_CONTAINER" \
  bash --noprofile --norc -lc '
    pgrep -af \
      "[p]ython .*tracer_ros1_joint_energy_proxy_logger_v0.py" \
      || true
  '

echo
echo "========== 7. unpause Gazebo physics =========="
docker exec a1_unitree_gazebo_docker bash --noprofile --norc -lc '
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
timeout 8s rosservice call /gazebo/unpause_physics "{}"
'

echo
echo "========== 8. quick status =========="
ros2 topic info /tracer/robot_odom_flat || true
ros2 topic info /tracer/terrain_context_label || true
ros2 topic info /tracer/mpc_reference || true

echo
echo "[TRACER] Phase-D4 context-meta stack is running."
echo "[TRACER] Monitor:"
echo "  tail -f $LOG_DIR/context_meta_selector.log"
echo "  tail -f $LOG_DIR/context_provider.log"
echo "  ros2 topic echo /tracer/terrain_context_label std_msgs/msg/String"
echo
echo "[TRACER] Stop:"
echo "  scripts/runtime/tracer_stop_phase_c_mixed_stack_v0.sh"

# Optional Gazebo GUI viewer. Run this at the end, after reset/bridges/policy are ready.
if [ "${TRACER_OPEN_GZCLIENT:-0}" = "1" ]; then
  echo
  echo "========== open Gazebo GUI gzclient =========="
  bash scripts/runtime/tracer_open_gzclient_v0.sh || true
fi
