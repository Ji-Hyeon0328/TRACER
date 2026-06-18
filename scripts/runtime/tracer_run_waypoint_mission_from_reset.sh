#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
DISTANCES="${TRACER_WAYPOINT_DISTANCES:-1.0,2.0,3.0}"
AUTO_STOP_QWERTY="${TRACER_AUTO_STOP_QWERTY:-0}"

echo "============================================================"
echo "[TRACER] Run waypoint mission from reset"
echo "============================================================"
echo "[TRACER] root:      $ROOT"
echo "[TRACER] distances: $DISTANCES"
echo

cd "$ROOT"

echo
echo "============================================================"
echo "[1/10] Reset to qwer state"
echo "============================================================"
scripts/runtime/tracer_reset_to_qwer_state.sh

echo
echo "============================================================"
echo "[2/10] Reset-state precheck"
echo "============================================================"
scripts/runtime/tracer_precheck_reset_state.sh || true

echo
echo "============================================================"
echo "[3/10] Start qwerty state"
echo "============================================================"
scripts/runtime/tracer_start_qwerty_state.sh

echo
echo "============================================================"
echo "[4/11] Ensure MPC reference bridge"
echo "============================================================"
scripts/runtime/tracer_ensure_mpc_ref_bridge.sh

echo
echo "============================================================"
echo "[4b/11] Set objective weights"
echo "============================================================"
OBJ_LABEL="${TRACER_OBJECTIVE_LABEL:-balanced}"
OBJ_WEIGHTS="${TRACER_OBJECTIVE_WEIGHTS:-0.3333333333333333,0.3333333333333333,0.3333333333333333}"
scripts/runtime/tracer_set_objective_weights.sh "$OBJ_LABEL" "$OBJ_WEIGHTS"

echo
echo "============================================================"
echo "[5/11] Qwerty live precheck"
echo "============================================================"
scripts/runtime/tracer_precheck_qwerty_state.sh

echo
echo "============================================================"
echo "[6/11] Start waypoint overlay"
echo "============================================================"
scripts/runtime/tracer_start_waypoint_overlay_from_qwerty.sh

echo
echo "============================================================"
echo "[7/11] Generate waypoints from current odom"
echo "============================================================"
scripts/runtime/tracer_generate_waypoints_from_current_odom.sh "$DISTANCES"

echo
echo "============================================================"
echo "[8/11] Waypoint mission live precheck"
echo "============================================================"
scripts/runtime/tracer_precheck_waypoint_mission.sh

echo
echo "============================================================"
echo "[9/11] Start mission logger"
echo "============================================================"
scripts/runtime/tracer_start_mission_logger.sh

echo
echo "============================================================"
echo "[10/11] Unpause mission"
echo "============================================================"
scripts/runtime/tracer_unpause_mission.sh

echo
echo "============================================================"
echo "[11/11] Wait for mission complete"
echo "============================================================"
if scripts/runtime/tracer_wait_for_mission_complete.sh; then
  MISSION_STATUS="success"
else
  MISSION_STATUS="timeout_or_failure"
fi

echo
echo "============================================================"
echo "[TRACER] Mission status: $MISSION_STATUS"
echo "============================================================"

echo
echo "[TRACER] Pausing mission after run..."
scripts/runtime/tracer_stop_mission.sh || true

echo
echo "[TRACER] Stopping logger and saving data..."
scripts/runtime/tracer_stop_mission_logger.sh || true

if [ "$AUTO_STOP_QWERTY" = "1" ]; then
  echo
  echo "[TRACER] Auto-stopping qwerty state..."
  scripts/runtime/tracer_stop_qwerty_state.sh || true
fi

echo
echo "============================================================"
echo "[TRACER] Run finished"
echo "============================================================"
echo "[TRACER] Mission status: $MISSION_STATUS"
echo
echo "[TRACER] Latest mission logs:"
ls -t "$ROOT"/data/mission_logs/tracer_mission_* 2>/dev/null | head -10 || true
echo
echo "[TRACER] Latest waypoint manager log:"
echo "  $(ls -td "$ROOT"/logs/waypoint_overlay_* 2>/dev/null | head -1)/waypoint_manager_v1.log"
echo
