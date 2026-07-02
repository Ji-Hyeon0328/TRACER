#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
TERRAIN="${1:-flat_normal}"

VX="${TRACER_OVERRIDE_VX:-0.09}"
YAW_RATE="${TRACER_OVERRIDE_YAW_RATE:-0.0}"
BODY_HEIGHT="${TRACER_OVERRIDE_BODY_HEIGHT:-0.32}"
SWING_CLEARANCE="${TRACER_OVERRIDE_SWING_CLEARANCE:-0.04}"
ENABLE="${TRACER_OVERRIDE_ENABLE:-1.0}"
HZ="${TRACER_OVERRIDE_PUBLISH_HZ:-20.0}"

LOG="/tmp/tracer_phase_a_override_policy_node_v0_${TERRAIN}.log"

cd "$ROOT"

echo "[TRACER] ensure Phase-A override policy node v0"
echo "[TRACER] root:      $ROOT"
echo "[TRACER] terrain:   $TERRAIN"
echo "[TRACER] vx:        $VX"
echo "[TRACER] yaw_rate:  $YAW_RATE"
echo "[TRACER] body_h:    $BODY_HEIGHT"
echo "[TRACER] clearance: $SWING_CLEARANCE"
echo "[TRACER] enable:    $ENABLE"
echo "[TRACER] hz:        $HZ"
echo "[TRACER] log:       $LOG"

echo
echo "========== source ROS2 =========="
set +u
source /opt/ros/humble/setup.bash
if [ -f "$ROOT/ros2_ws/install/setup.bash" ]; then
  export COLCON_TRACE="${COLCON_TRACE:-}"
  source "$ROOT/ros2_ws/install/setup.bash"
fi
set -u

echo
echo "========== stop old Phase-A policy nodes =========="
pkill -9 -f tracer_phase_a_ram_aware_policy_node_v0.py 2>/dev/null || true
pkill -9 -f tracer_phase_a_override_policy_node_v0.py 2>/dev/null || true
sleep 0.5

echo
echo "========== refresh ROS2 daemon after kill =========="
ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1 || true
sleep 0.5

echo
echo "========== wait for ROS2 publisher graph to clear =========="
for i in $(seq 1 10); do
  CNT="$(ros2 topic info /tracer/mpc_reference 2>/dev/null | awk '/Publisher count:/ {print $3}' || echo 0)"
  CNT="${CNT:-0}"
  echo "[TRACER] wait graph clear attempt=$i publisher_count=$CNT"
  if [ "$CNT" = "0" ]; then
    break
  fi
  sleep 0.5
done

echo
echo "========== start override policy node =========="
nohup /usr/bin/python3 \
  "$ROOT/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/tracer_phase_a_override_policy_node_v0.py" \
  --ros-args \
  -p terrain:="$TERRAIN" \
  -p publish_hz:="$HZ" \
  -p vx:="$VX" \
  -p yaw_rate:="$YAW_RATE" \
  -p body_height:="$BODY_HEIGHT" \
  -p swing_clearance:="$SWING_CLEARANCE" \
  -p enable:="$ENABLE" \
  > "$LOG" 2>&1 &

sleep 1.0

echo
echo "========== process check =========="
pgrep -af "tracer_phase_a_override_policy_node_v0.py|tracer_phase_a_ram_aware_policy_node_v0.py" || true

echo
echo "========== log tail =========="
tail -20 "$LOG" || true

echo
echo "========== ROS2 /tracer/mpc_reference topic info =========="
ros2 topic info -v /tracer/mpc_reference || true

echo
echo "========== one-shot status =========="
timeout 3 ros2 topic echo --once /tracer/mpc_reference || true
timeout 3 ros2 topic echo --once /tracer/highlevel_debug || true
