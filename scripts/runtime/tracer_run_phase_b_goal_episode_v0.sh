#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
GAZEBO_CONTAINER="${TRACER_GAZEBO_CONTAINER:-a1_unitree_gazebo_docker}"
CTRL_CONTAINER="${TRACER_CTRL_CONTAINER:-a1_cpp_ctrl_docker}"
MODEL_NAME="${TRACER_GAZEBO_MODEL_NAME:-a1_gazebo}"

WORLD_NAME="${TRACER_PHASE_B_WORLD:-earth}"
DURATION="${TRACER_PHASE_B_RECORD_DURATION:-10.0}"
SAMPLE_HZ="${TRACER_PHASE_B_SAMPLE_HZ:-20.0}"
STOP_DISTANCE="${TRACER_PHASE_B_GOAL_STOP_DISTANCE:-0.15}"
GOAL_DISTANCE="${TRACER_PHASE_B_GOAL_DISTANCE_AHEAD:-0.5}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${TRACER_PHASE_B_RUN_DIR:-$ROOT/artifacts/phase_b_goal_episode_${STAMP}}"
CSV_PATH="$RUN_DIR/phase_b_episode.csv"
SUMMARY_PATH="$RUN_DIR/phase_b_summary.json"

mkdir -p "$RUN_DIR"

echo "[TRACER] Phase-B goal episode runner"
echo "[TRACER] root:      $ROOT"
echo "[TRACER] world:     $WORLD_NAME"
echo "[TRACER] model:     $MODEL_NAME"
echo "[TRACER] run_dir:   $RUN_DIR"
echo "[TRACER] duration:  $DURATION"
echo "[TRACER] goal_dist: $GOAL_DISTANCE"

cd "$ROOT"

function ros1_ctrl() {
  docker exec "$CTRL_CONTAINER" bash -lc "
    set +u
    source /opt/ros/melodic/setup.bash
    source /root/unitree_ws/devel/setup.bash 2>/dev/null || true
    source /root/A1_ctrl_ws/devel/setup.bash 2>/dev/null || true
    $*
  "
}

function ensure_world() {
  echo
  echo "========== verify world =========="
  if docker exec "$GAZEBO_CONTAINER" bash -lc "ps aux | grep -E 'gzserver|normal.launch|${WORLD_NAME}' | grep -v grep | grep -q '${WORLD_NAME}'"; then
    echo "[TRACER] world already running: $WORLD_NAME"
  else
    echo "[TRACER] world not running as $WORLD_NAME. launching clean world..."
    "$ROOT/scripts/runtime/tracer_launch_gazebo_world_clean.sh" "$WORLD_NAME"
  fi

  docker exec "$GAZEBO_CONTAINER" bash -lc "
    ps aux | grep -E 'gzserver|normal.launch|earth|slippery|sponge|rough|slope' | grep -v grep || true
  "
}

function ensure_mpc_ref_bridge() {
  echo
  echo "========== ensure ROS2<->ROS1 mpc_reference bridge =========="
  set +u
  source /opt/ros/humble/setup.bash
  set -u

  pkill -f "tracer_ros2_mpc_ref_udp_sender.py" 2>/dev/null || true

  docker exec "$CTRL_CONTAINER" bash -lc '
    pkill -f "tracer_udp_to_ros1_mpc_ref.py" 2>/dev/null || true
  ' || true

  sleep 0.5

  docker exec -d "$CTRL_CONTAINER" bash --noprofile --norc -lc '
    set +u
    source /opt/ros/melodic/setup.bash
    source /root/unitree_ws/devel/setup.bash 2>/dev/null || true
    source /root/A1_ctrl_ws/devel/setup.bash 2>/dev/null || true
    rosrun a1_cpp tracer_udp_to_ros1_mpc_ref.py \
      > /tmp/tracer_udp_to_ros1_mpc_ref.log 2>&1
  '

  nohup /usr/bin/python3 \
    "$ROOT/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_ros2_mpc_ref_udp_sender.py" \
    > /tmp/tracer_ros2_mpc_ref_udp_sender.log 2>&1 &

  echo $! > /tmp/tracer_ros2_mpc_ref_udp_sender.pid

  sleep 1.0
}

function print_bridge_status() {
  echo
  echo "========== bridge status =========="
  set +u
  source /opt/ros/humble/setup.bash
  set -u

  echo "[ROS2 /tracer/mpc_reference]"
  ros2 topic info -v /tracer/mpc_reference || true

  echo
  echo "[ROS1 /tracer/mpc_reference]"
  ros1_ctrl "rostopic info /tracer/mpc_reference || true"
}

function cleanup_end() {
  if [[ "${TRACER_PHASE_B_PAUSE_AT_END:-1}" == "1" ]]; then
    echo
    echo "========== pause physics =========="
    ros1_ctrl "rosservice call /gazebo/pause_physics '{}' || true" || true
  fi
}
trap cleanup_end EXIT

ensure_world

echo
echo "========== hard reset =========="
TRACER_GAZEBO_MODEL_NAME="$MODEL_NAME" \
"$ROOT/scripts/runtime/tracer_gazebo_hard_reset_a1_stand.sh"

echo
echo "========== odom bridge =========="
TRACER_GAZEBO_MODEL_NAME="$MODEL_NAME" \
"$ROOT/scripts/runtime/tracer_start_modelstate_odom_bridge_only.sh"

echo
echo "========== unpause briefly for odom =========="
ros1_ctrl "rosservice call /gazebo/unpause_physics '{}'"

sleep 0.5

echo
echo "========== wait/check odom =========="
set +u
source /opt/ros/humble/setup.bash
set -u
timeout 3s ros2 topic echo --once /tracer/robot_odom_flat || true

echo
echo "========== start Phase-B policy =========="
TRACER_PHASE_B_VX_FAR="${TRACER_PHASE_B_VX_FAR:-0.09}" \
TRACER_PHASE_B_VX_NEAR="${TRACER_PHASE_B_VX_NEAR:-0.04}" \
TRACER_PHASE_B_BODY_HEIGHT="${TRACER_PHASE_B_BODY_HEIGHT:-0.32}" \
TRACER_PHASE_B_SWING_CLEARANCE="${TRACER_PHASE_B_SWING_CLEARANCE:-0.04}" \
TRACER_PHASE_B_GOAL_STOP_DISTANCE="${TRACER_PHASE_B_GOAL_STOP_DISTANCE:-0.15}" \
TRACER_PHASE_B_GOAL_SLOW_DISTANCE="${TRACER_PHASE_B_GOAL_SLOW_DISTANCE:-0.25}" \
TRACER_PHASE_B_RELATIVE_STOP_RADIUS="${TRACER_PHASE_B_RELATIVE_STOP_RADIUS:-0.0}" \
TRACER_PHASE_B_GOAL_TIMEOUT_SEC="${TRACER_PHASE_B_GOAL_TIMEOUT_SEC:-60.0}" \
"$ROOT/scripts/runtime/tracer_ensure_goal_meta_policy_node_v0.sh"

ensure_mpc_ref_bridge
print_bridge_status

echo
echo "========== start recorder =========="
/usr/bin/python3 "$ROOT/scripts/runtime/tracer_phase_b_episode_recorder_v0.py" \
  --duration "$DURATION" \
  --sample-hz "$SAMPLE_HZ" \
  --stop-distance "$STOP_DISTANCE" \
  --csv-path "$CSV_PATH" \
  --summary-path "$SUMMARY_PATH" \
  > "$RUN_DIR/recorder_stdout.log" 2>&1 &

REC_PID=$!

sleep 1.0

echo
echo "========== publish goal ahead =========="
timeout 4s /usr/bin/python3 \
  "$ROOT/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_goal_ahead_publisher_node.py" \
  --ros-args \
  -p distance_ahead:="$GOAL_DISTANCE" \
  -p one_shot:=true \
  -p publish_hz:=5.0 \
  > "$RUN_DIR/goal_ahead.log" 2>&1 || true

echo "[TRACER] waiting recorder pid=$REC_PID"
if ! wait "$REC_PID"; then
  echo "[TRACER][ERROR] recorder failed. recorder_stdout.log:"
  sed -n '1,200p' "$RUN_DIR/recorder_stdout.log" || true
  exit 1
fi

if [[ ! -s "$SUMMARY_PATH" ]]; then
  echo "[TRACER][ERROR] summary file missing: $SUMMARY_PATH"
  echo "[TRACER][ERROR] recorder_stdout.log:"
  sed -n '1,200p' "$RUN_DIR/recorder_stdout.log" || true
  exit 1
fi

echo
echo "========== episode summary =========="
cat "$SUMMARY_PATH"

echo
echo "========== quick final topic samples =========="
echo "[relative_goal]"
timeout 2s ros2 topic echo --once /tracer/relative_goal || true

echo "[mpc_reference]"
timeout 2s ros2 topic echo --once /tracer/mpc_reference || true

echo "[ROS1 mpc_reference]"
ros1_ctrl "timeout 2s rostopic echo -n 1 /tracer/mpc_reference || true"

echo
echo "[TRACER] outputs:"
echo "  CSV:     $CSV_PATH"
echo "  Summary: $SUMMARY_PATH"
echo "  Logs:    $RUN_DIR"
