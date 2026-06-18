#!/usr/bin/env bash
set -eo pipefail

# Save terminal state because docker/ROS background shutdown can occasionally
# leave the interactive terminal in a broken line discipline state.
ORIG_STTY="$(stty -g 2>/dev/null || true)"

ROOT="${TRACER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"

STYLE_CFG="${TRACER_STYLE_CONFIG:-configs/style_presets/style_presets_v0.yaml}"
STYLE_NAME="${TRACER_STYLE_NAME:-nominal}"
TERRAIN_NAME="${TRACER_TERRAIN_NAME:-unknown}"
WORLD_NAME="${TRACER_WORLD_NAME:-unknown}"
WAYPOINT_DISTANCES="${TRACER_WAYPOINT_DISTANCES:-1.0,2.0,3.0}"
TIMEOUT_SEC="${TRACER_MISSION_TIMEOUT_SEC:-60}"
STYLE_HZ="${TRACER_STYLE_HZ:-20}"
STYLE_DURATION_SEC="${TRACER_STYLE_DURATION_SEC:-$((TIMEOUT_SEC + 20))}"
RUN_ID="${TRACER_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

echo "============================================================"
echo "[TRACER] Run style preset mission"
echo "============================================================"
echo "[TRACER] root:      $ROOT"
echo "[TRACER] style cfg: $STYLE_CFG"
echo "[TRACER] style:     $STYLE_NAME"
echo "[TRACER] terrain:   $TERRAIN_NAME"
echo "[TRACER] world:     $WORLD_NAME"
echo "[TRACER] distances: $WAYPOINT_DISTANCES"
echo "[TRACER] timeout:   $TIMEOUT_SEC"

eval "$(python3 scripts/runtime/tracer_style_preset_env.py --config "$STYLE_CFG" --style "$STYLE_NAME")"

echo
echo "[TRACER] loaded style preset:"
echo "  name:        $TRACER_STYLE_NAME"
echo "  vx:          $TRACER_STYLE_VX"
echo "  yaw_rate:    $TRACER_STYLE_YAW_RATE"
echo "  body_height: $TRACER_STYLE_BODY_HEIGHT"
echo "  clearance:   $TRACER_STYLE_CLEARANCE"
echo "  enable:      $TRACER_STYLE_ENABLE"
echo "  beta_hint:   $TRACER_STYLE_BETA_HINT"

OUT_DIR="$ROOT/data/style_missions/${RUN_ID}_${STYLE_NAME}"
LOG_DIR="$ROOT/logs/style_mission_${RUN_ID}_${STYLE_NAME}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

STYLE_PUB_PID=""


restore_terminal() {
  if [ -n "${ORIG_STTY:-}" ]; then
    stty "$ORIG_STTY" 2>/dev/null || stty sane 2>/dev/null || true
  else
    stty sane 2>/dev/null || true
  fi
}

cleanup() {
  set +e
  echo
  echo "[TRACER] cleanup style mission"
  if [ -n "$STYLE_PUB_PID" ]; then
    # Try to stop the whole background publisher process tree.
    kill -INT "$STYLE_PUB_PID" 2>/dev/null || true
    sleep 0.2
    kill -TERM "$STYLE_PUB_PID" 2>/dev/null || true
    wait "$STYLE_PUB_PID" 2>/dev/null || true
  fi

  # In case a ROS1 Python publisher remains inside the controller container.
  sudo docker exec "${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}" bash --noprofile --norc -lc '
  pkill -f "tracer_style_mpc_ref_ros1_publisher" || true
  ' >/dev/null 2>&1 || true


  TRACER_STYLE_NAME=stop \
  TRACER_STYLE_VX=0.0 \
  TRACER_STYLE_YAW_RATE=0.0 \
  TRACER_STYLE_BODY_HEIGHT=0.30 \
  TRACER_STYLE_CLEARANCE=0.03 \
  TRACER_STYLE_ENABLE=0.0 \
  TRACER_STYLE_DURATION_SEC=2 \
  scripts/runtime/tracer_publish_style_mpc_ref_ros1.sh >/tmp/tracer_style_stop.log 2>&1 || true

  scripts/runtime/tracer_stop_mission.sh >/tmp/tracer_style_stop_mission.log 2>&1 || true

  if [ -n "$ORIG_STTY" ]; then
    stty "$ORIG_STTY" 2>/dev/null || true
  else
    stty sane 2>/dev/null || true
  fi
  restore_terminal
}
trap cleanup EXIT

echo
echo "============================================================"
echo "[1/10] Reset to qwer state"
echo "============================================================"
scripts/runtime/tracer_robust_reset_to_qwer_state.sh

echo
echo "============================================================"
echo "[2/10] Reset-state precheck"
echo "============================================================"
scripts/runtime/tracer_precheck_reset_state.sh

echo
echo "============================================================"
echo "[3/10] Start qwerty state"
echo "============================================================"
scripts/runtime/tracer_start_qwerty_state.sh

echo
echo "============================================================"
echo "[4/10] Stop learned MPC reference path"
echo "============================================================"
scripts/runtime/tracer_stop_learned_mpc_ref_path.sh

echo
echo "============================================================"
echo "[5/10] Ensure MPC reference bridge/subscriber"
echo "============================================================"
scripts/runtime/tracer_ensure_mpc_ref_bridge.sh || true

echo
echo "[TRACER] ROS1 /tracer/mpc_reference topic info after learned path stop:"
sudo docker exec "${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rostopic info /tracer/mpc_reference || true
'

echo
echo "[TRACER] require /gazebo_a1_ctrl subscriber on /tracer/mpc_reference"
if ! sudo docker exec "${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}" bash --noprofile --norc -lc '
source /opt/ros/melodic/setup.bash
source /root/unitree_ws/devel/setup.bash
source /root/A1_ctrl_ws/devel/setup.bash
rostopic info /tracer/mpc_reference 2>/dev/null | grep -q "/gazebo_a1_ctrl"
'; then
  echo "[ERROR] /gazebo_a1_ctrl is not subscribed to /tracer/mpc_reference."
  echo "[ERROR] Start or restart the A1-QP-MPC controller first:"
  echo "        scripts/runtime/tracer_start_a1_qpmc_controller_only.sh"
  exit 20
fi
echo "[TRACER] /gazebo_a1_ctrl subscriber confirmed."

echo
echo "============================================================"
echo "[5b/10] Publish beta_hint as objective_weights metadata"
echo "============================================================"
if [ -x scripts/runtime/tracer_set_objective_weights.sh ]; then
  TRACER_OBJECTIVE_LABEL="$STYLE_NAME" \
  TRACER_OBJECTIVE_WEIGHTS="$TRACER_STYLE_BETA_HINT" \
  scripts/runtime/tracer_set_objective_weights.sh || true
else
  echo "[TRACER] tracer_set_objective_weights.sh not found; skip."
fi

echo
echo "============================================================"
echo "[6/10] Start waypoint overlay"
echo "============================================================"
scripts/runtime/tracer_start_waypoint_overlay_from_qwerty.sh

echo
echo "============================================================"
echo "[7/10] Generate waypoints from current odom"
echo "============================================================"
export TRACER_WAYPOINT_DISTANCES="$WAYPOINT_DISTANCES"
scripts/runtime/tracer_generate_waypoints_from_current_odom.sh "$WAYPOINT_DISTANCES"

WAYPOINT_LOG="$(ls -td "$ROOT"/logs/waypoint_overlay_* 2>/dev/null | head -1)/waypoint_manager_v1.log"
echo "[TRACER] waypoint log: $WAYPOINT_LOG"

echo
echo "============================================================"
echo "[8/10] Start mission logger"
echo "============================================================"
scripts/runtime/tracer_start_mission_logger.sh

echo
echo "============================================================"
echo "[9/10] Start style publisher and unpause mission"
echo "============================================================"
TRACER_STYLE_HZ="$STYLE_HZ" \
TRACER_STYLE_DURATION_SEC="$STYLE_DURATION_SEC" \
scripts/runtime/tracer_publish_style_mpc_ref_ros1.sh > "$LOG_DIR/style_publisher.log" 2>&1 < /dev/null &
STYLE_PUB_PID=$!

sleep 1

scripts/runtime/tracer_unpause_mission.sh

echo
echo "============================================================"
echo "[10/10] Wait for mission complete"
echo "============================================================"
echo "[TRACER] timeout: $TIMEOUT_SEC sec"

MISSION_STATUS="timeout"
START_TS="$(date +%s)"

while true; do
  if [ -f "$WAYPOINT_LOG" ] && grep -q "mission complete" "$WAYPOINT_LOG"; then
    MISSION_STATUS="success"
    echo "[TRACER] mission complete detected."
    break
  fi

  NOW_TS="$(date +%s)"
  ELAPSED=$((NOW_TS - START_TS))
  if [ "$ELAPSED" -ge "$TIMEOUT_SEC" ]; then
    MISSION_STATUS="timeout"
    echo "[TRACER] mission timeout."
    break
  fi

  sleep 1
done

echo
echo "============================================================"
echo "[TRACER] Mission status: $MISSION_STATUS"
echo "============================================================"

echo
echo "[TRACER] stop style publisher"
if [ -n "$STYLE_PUB_PID" ]; then
  kill "$STYLE_PUB_PID" 2>/dev/null || true
  wait "$STYLE_PUB_PID" 2>/dev/null || true
  STYLE_PUB_PID=""
fi

echo
echo "[TRACER] publish stop command"
TRACER_STYLE_NAME=stop \
TRACER_STYLE_VX=0.0 \
TRACER_STYLE_YAW_RATE=0.0 \
TRACER_STYLE_BODY_HEIGHT=0.30 \
TRACER_STYLE_CLEARANCE=0.03 \
TRACER_STYLE_ENABLE=0.0 \
TRACER_STYLE_DURATION_SEC=2 \
scripts/runtime/tracer_publish_style_mpc_ref_ros1.sh > "$LOG_DIR/stop_publisher.log" 2>&1 || true

echo
echo "[TRACER] stop mission and logger"
scripts/runtime/tracer_stop_mission.sh || true
scripts/runtime/tracer_stop_mission_logger.sh || true

LATEST_SUMMARY="$(ls -t "$ROOT"/data/mission_logs/tracer_mission_summary_*.json 2>/dev/null | head -1 || true)"
LATEST_NPZ="$(ls -t "$ROOT"/data/mission_logs/tracer_mission_log_*.npz 2>/dev/null | head -1 || true)"

if [ -n "$LATEST_SUMMARY" ]; then
  cp "$LATEST_SUMMARY" "$OUT_DIR/mission_summary.json"
fi
if [ -n "$LATEST_NPZ" ]; then
  cp "$LATEST_NPZ" "$OUT_DIR/mission_log.npz"
fi
if [ -f "$WAYPOINT_LOG" ]; then
  cp "$WAYPOINT_LOG" "$OUT_DIR/waypoint_manager.log"
fi
cp "$LOG_DIR/style_publisher.log" "$OUT_DIR/style_publisher.log" 2>/dev/null || true

python3 - <<PY
import json
from pathlib import Path

meta = {
    "run_id": "$RUN_ID",
    "mission_status": "$MISSION_STATUS",
    "style_config": "$STYLE_CFG",
    "style_name": "$TRACER_STYLE_NAME",
    "terrain_name": "$TERRAIN_NAME",
    "world_name": "$WORLD_NAME",
    "style_command": {
        "vx": float("$TRACER_STYLE_VX"),
        "yaw_rate": float("$TRACER_STYLE_YAW_RATE"),
        "body_height": float("$TRACER_STYLE_BODY_HEIGHT"),
        "swing_clearance": float("$TRACER_STYLE_CLEARANCE"),
        "enable": float("$TRACER_STYLE_ENABLE"),
    },
    "beta_hint": [float(x) for x in "$TRACER_STYLE_BETA_HINT".split(",")],
    "waypoint_distances": "$WAYPOINT_DISTANCES",
    "timeout_sec": float("$TIMEOUT_SEC"),
    "latest_summary": "$LATEST_SUMMARY",
    "latest_npz": "$LATEST_NPZ",
}
out = Path("$OUT_DIR") / "style_mission_metadata.json"
out.write_text(json.dumps(meta, indent=2))
print("[TRACER] wrote", out)
PY

echo
echo "[TRACER] output dir: $OUT_DIR"
echo "[TRACER] style publisher log:"
tail -30 "$OUT_DIR/style_publisher.log" 2>/dev/null || true

echo
echo "[TRACER] callback tail:"
sudo docker exec "${TRACER_A1_CONTAINER:-a1_cpp_ctrl_docker}" bash --noprofile --norc -lc '
grep -R "\[TRACER\] mpc_reference" \
  /tmp/tracer_a1_ctrl.launch.log /root/.ros/log -n 2>/dev/null | tail -20 || true
'

trap - EXIT
cleanup >/dev/null 2>&1 || true

echo
restore_terminal

echo "============================================================"
echo "[TRACER] Style preset mission finished"
echo "============================================================"
echo "[TRACER] status:  $MISSION_STATUS"
if [ -n "$ORIG_STTY" ]; then
  stty "$ORIG_STTY" 2>/dev/null || stty sane 2>/dev/null || true
fi

echo "[TRACER] out dir: $OUT_DIR"
