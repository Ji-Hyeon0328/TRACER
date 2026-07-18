#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WORLD="${TRACER_PHASE_C_WORLD:-tracer_mixed_solid_course_v0}"
GOAL_X="${TRACER_PHASE_C_GOAL_X:-8.0}"
export TRACER_PHASE_C_GOAL_X="$GOAL_X"
PUB_HZ="${TRACER_PHASE_C_PUB_HZ:-10}"
VX="${TRACER_PHASE_C_FIXED_VX:-0.16}"
BODY_H="${TRACER_PHASE_C_FIXED_BODY_H:-0.32}"
CLEARANCE="${TRACER_PHASE_C_FIXED_CLEARANCE:-0.045}"
LOG_DIR="$ROOT/logs/phase_d_segment_action_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "[TRACER] start Phase-D segment-action stack v0"
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
scripts/runtime/tracer_launch_gazebo_world_clean.sh "$WORLD"


echo
echo "========== 2. qwer standing reset =========="
scripts/runtime/tracer_reset_to_qwer_state.sh

echo
echo "========== 3. validate standing pose =========="
scripts/runtime/tracer_validate_a1_ready_standing_pose.sh

echo
echo "

========== 4. start bridges =========="
scripts/runtime/tracer_ensure_ros2_mpc_ref_udp_sender.sh
scripts/runtime/tracer_ensure_mpc_ref_bridge.sh
scripts/runtime/tracer_start_modelstate_odom_bridge_only.sh

echo
echo "========== 5. start mixed terrain context provider =========="
nohup /usr/bin/python3 \
  scripts/runtime/tracer_mixed_terrain_context_provider_v0.py \
  > "$LOG_DIR/context_provider.log" 2>&1 &
echo $! > "$LOG_DIR/context_provider.pid"
sleep 1.0

echo
echo "========== 6. start segment-action MPC ref node =========="
nohup /usr/bin/python3 \
  scripts/runtime/tracer_phase_d_segment_action_mpc_ref_node_v0.py \
  --goal-x "$GOAL_X" \
  --pub-hz "$PUB_HZ" \
  --vx "$VX" \
  --body-height "$BODY_H" \
  --clearance "$CLEARANCE" \
  > "$LOG_DIR/segment_action.log" 2>&1 &
echo $! > "$LOG_DIR/segment_action.pid"
sleep 1.0

echo
echo "========== 7. unpause Gazebo physics =========="
docker exec a1_unitree_gazebo_docker bash --noprofile --norc -lc '
set +u
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
rosservice call /gazebo/unpause_physics "{}"
'

echo
echo "========== 8. quick status =========="
ros2 topic info /tracer/robot_odom_flat || true
ros2 topic info /tracer/terrain_context_label || true
ros2 topic info /tracer/mpc_reference || true

echo
echo "[TRACER] Phase-D segment-action stack is running."
echo "[TRACER] Monitor:"
echo "  tail -f $LOG_DIR/segment_action.log"
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

