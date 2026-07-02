#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
TERRAIN="${1:-sponge_firm_flat}"
HZ="${TRACER_PHASE_A_HZ:-20.0}"
REQUIRE_FINAL_GUARD="${TRACER_REQUIRE_FINAL_GUARD:-true}"
LOG="/tmp/tracer_phase_a_ram_aware_policy_node_v0_${TERRAIN}.log"

HOLD_VX="${TRACER_PHASE_A_HOLD_VX:-0.0}"
HOLD_YAW_RATE="${TRACER_PHASE_A_HOLD_YAW_RATE:-0.0}"
HOLD_BODY_HEIGHT="${TRACER_PHASE_A_HOLD_BODY_HEIGHT:-0.30}"
HOLD_SWING_CLEARANCE="${TRACER_PHASE_A_HOLD_SWING_CLEARANCE:-0.03}"
HOLD_ENABLE="${TRACER_PHASE_A_HOLD_ENABLE:-1.0}"

cd "$ROOT"

echo "[TRACER] ensure Phase-A RAM-aware policy node v0"
echo "[TRACER] root:    $ROOT"
echo "[TRACER] terrain: $TERRAIN"
echo "[TRACER] hz:      $HZ"
echo "[TRACER] log:     $LOG"
echo "[TRACER] hold_vx:     $HOLD_VX"
echo "[TRACER] hold_yaw:    $HOLD_YAW_RATE"
echo "[TRACER] hold_body_h: $HOLD_BODY_HEIGHT"
echo "[TRACER] hold_clr:    $HOLD_SWING_CLEARANCE"
echo "[TRACER] hold_enable: $HOLD_ENABLE"

echo
echo "========== source ROS2 =========="
set +u
export COLCON_TRACE="${COLCON_TRACE:-}"
source /opt/ros/humble/setup.bash
if [ -f "$ROOT/ros2_ws/install/setup.bash" ]; then
  source "$ROOT/ros2_ws/install/setup.bash"
fi
set -u

echo
echo "========== stop old Phase-A RAM-aware policy nodes =========="
for i in $(seq 1 5); do
  OLD_PIDS="$(pgrep -f 'tracer_phase_a_ram_aware_policy_node_v0.py' || true)"
  if [ -z "$OLD_PIDS" ]; then
    break
  fi
  echo "[TRACER] killing old pids: $OLD_PIDS"
  pkill -9 -f 'tracer_phase_a_ram_aware_policy_node_v0.py' 2>/dev/null || true
pkill -9 -f tracer_phase_a_override_policy_node_v0.py 2>/dev/null || true
  sleep 1
done

echo
echo "========== refresh ROS2 daemon after kill =========="
ros2 daemon stop >/dev/null 2>&1 || true
sleep 1
ros2 daemon start >/dev/null 2>&1 || true
sleep 1

echo
echo "========== wait for ROS2 publisher graph to clear =========="
for i in $(seq 1 5); do
  PUB_COUNT="$(ros2 topic info /tracer/mpc_reference 2>/dev/null | awk '/Publisher count:/ {print $3}' || true)"
  PUB_COUNT="${PUB_COUNT:-0}"

  echo "[TRACER] wait graph clear attempt=$i publisher_count=$PUB_COUNT"

  if [ "$PUB_COUNT" = "0" ]; then
    break
  fi

  sleep 1
done

OLD_PIDS_AFTER_WAIT="$(pgrep -f 'tracer_phase_a_ram_aware_policy_node_v0.py' || true)"
PUB_COUNT_AFTER_WAIT="$(ros2 topic info /tracer/mpc_reference 2>/dev/null | awk '/Publisher count:/ {print $3}' || true)"
PUB_COUNT_AFTER_WAIT="${PUB_COUNT_AFTER_WAIT:-0}"

if [ -z "$OLD_PIDS_AFTER_WAIT" ] && [ "$PUB_COUNT_AFTER_WAIT" != "0" ]; then
  echo "[TRACER][WARN] No old Phase-A process remains, but ROS2 graph still reports publisher_count=$PUB_COUNT_AFTER_WAIT."
  echo "[TRACER][WARN] Treating this as stale DDS/ros2cli graph and continuing."
fi

echo
echo "========== process check before start =========="
pgrep -af 'tracer_phase_a_ram_aware_policy_node_v0.py' || true

echo
echo "========== start new Phase-A RAM-aware policy node =========="
nohup /usr/bin/python3 \
  "$ROOT/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_phase_a_ram_aware_policy_node_v0.py" \
  --ros-args \
  -p tracer_root:="$ROOT" \
  -p terrain:="$TERRAIN" \
  -p publish_hz:="$HZ" \
  -p require_final_guard:="$REQUIRE_FINAL_GUARD" \
  -p hold_vx:="$HOLD_VX" \
  -p hold_yaw_rate:="$HOLD_YAW_RATE" \
  -p hold_body_height:="$HOLD_BODY_HEIGHT" \
  -p hold_swing_clearance:="$HOLD_SWING_CLEARANCE" \
  -p hold_enable:="$HOLD_ENABLE" \
  > "$LOG" 2>&1 &

sleep 1

echo
echo "========== process check =========="
pgrep -af 'tracer_phase_a_ram_aware_policy_node_v0.py' || true

echo
echo "========== log tail =========="
tail -n 30 "$LOG" || true

echo
echo "========== ROS2 /tracer/mpc_reference topic info =========="
ros2 topic info -v /tracer/mpc_reference || true

echo
echo "========== one-shot status =========="
timeout 3 ros2 topic echo --once /tracer/mpc_reference || true
timeout 3 ros2 topic echo --once /tracer/phase_a_policy_status || true
