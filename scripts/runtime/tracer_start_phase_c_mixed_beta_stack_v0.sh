#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WORLD="${TRACER_PHASE_C_WORLD:-tracer_mixed_solid_course_v0}"
GOAL_X="${TRACER_PHASE_C_GOAL_X:-8.0}"
PUB_HZ="${TRACER_PHASE_C_PUB_HZ:-10}"
LOG_DIR="$ROOT/logs/phase_c_mixed_beta_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "[TRACER] start Phase-C mixed beta-aware stack v0"
echo "[TRACER] world:   $WORLD"
echo "[TRACER] goal_x:  $GOAL_X"
echo "[TRACER] pub_hz:  $PUB_HZ"
echo "[TRACER] logs:    $LOG_DIR"

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
echo "========== 4. start bridges =========="
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
echo "========== 6. start beta-aware MPC ref node =========="
nohup /usr/bin/python3 \
  scripts/runtime/tracer_phase_c_beta_schedule_mpc_ref_node_v0.py \
  --goal-x "$GOAL_X" \
  --pub-hz "$PUB_HZ" \
  > "$LOG_DIR/beta_schedule.log" 2>&1 &
echo $! > "$LOG_DIR/beta_schedule.pid"
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
ros2 topic info /tracer/objective_weights || true
ros2 topic info /tracer/mpc_reference || true

echo
echo "[TRACER] Phase-C mixed beta-aware stack is running."
echo "[TRACER] Monitor:"
echo "  tail -f $LOG_DIR/beta_schedule.log"
echo "  tail -f $LOG_DIR/context_provider.log"
echo "  ros2 topic echo /tracer/objective_weights std_msgs/msg/Float64MultiArray"
echo
echo "[TRACER] Stop:"
echo "  scripts/runtime/tracer_stop_phase_c_mixed_stack_v0.sh"
