#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
TERRAIN="${1:-sponge_firm_flat}"
WORLD="${2:-tracer_sponge_firm_flat}"
POLICY_ID="${3:-phase_a_ram_aware_runtime_debug_v1_pause_ready_unpause}"

cd "$ROOT"

echo "[TRACER] run Phase-A runtime episode v0"
echo "[TRACER] terrain:   $TERRAIN"
echo "[TRACER] world:     $WORLD"
echo "[TRACER] policy_id: $POLICY_ID"

echo
echo "========== 1. launch world/controller =========="
scripts/runtime/tracer_launch_gazebo_world_clean.sh "$WORLD"

echo
echo "========== 2. hard reset to stand, keep physics paused =========="
scripts/runtime/tracer_gazebo_hard_reset_a1_stand.sh || true

echo
echo "========== 3. start bridges while physics is paused =========="
scripts/runtime/tracer_start_data_collection_lite.sh

echo
echo "========== 4. start Phase-A runtime policy =========="
if [ "${TRACER_PHASE_A_USE_OVERRIDE:-0}" = "1" ]; then
  scripts/runtime/tracer_ensure_phase_a_override_policy_node_v0.sh "$TERRAIN"
else
  scripts/runtime/tracer_ensure_phase_a_ram_aware_policy_node_v0.sh "$TERRAIN"
fi

echo
echo "========== 5. wait/verify command/debug before unpause =========="

if [ -n "${TRACER_PHASE_A_EXPECT_MODE:-}" ]; then
  WAIT_ARGS=(--expect-mode "$TRACER_PHASE_A_EXPECT_MODE")

  if [ -n "${TRACER_PHASE_A_EXPECT_VX:-}" ]; then
    WAIT_ARGS+=(--expect-vx "$TRACER_PHASE_A_EXPECT_VX")
  fi
  if [ -n "${TRACER_PHASE_A_EXPECT_BODY_HEIGHT:-}" ]; then
    WAIT_ARGS+=(--expect-body-height "$TRACER_PHASE_A_EXPECT_BODY_HEIGHT")
  fi
  if [ -n "${TRACER_PHASE_A_EXPECT_SWING_CLEARANCE:-}" ]; then
    WAIT_ARGS+=(--expect-clearance "$TRACER_PHASE_A_EXPECT_SWING_CLEARANCE")
  fi

  /usr/bin/python3 scripts/runtime/tracer_wait_phase_a_policy_status_v0.py \
    "${WAIT_ARGS[@]}" \
    --timeout-sec "${TRACER_PHASE_A_EXPECT_TIMEOUT_SEC:-8.0}"
fi

ros2 topic info -v /tracer/mpc_reference || true
timeout 3 ros2 topic echo --once /tracer/mpc_reference || true
timeout 3 ros2 topic echo --once /tracer/highlevel_debug || true

echo
echo "========== 6. unpause physics =========="
docker exec -i a1_unitree_gazebo_docker bash -lc '
  set +u
  source /opt/ros/melodic/setup.bash
  rosservice call /gazebo/unpause_physics "{}" || true
'

sleep "${TRACER_RUNTIME_POST_UNPAUSE_SLEEP:-1}"

echo
echo "========== 7. record runtime rollout =========="
scripts/runtime/tracer_record_phase_a_runtime_rollout_v0.sh "$TERRAIN" "$POLICY_ID"

echo
echo "[TRACER] Phase-A runtime episode completed"
