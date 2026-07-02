#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
TERRAIN="${1:-sponge_firm_flat}"
POLICY_ID="${2:-phase_a_ram_aware_runtime_v0}"
DURATION_SEC="${TRACER_RUNTIME_ROLLOUT_DURATION_SEC:-5.0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT/data/rollouts/phase_a_runtime_v0/runtime_rollout_${TS}"
EPISODE_ID="${TERRAIN}_${POLICY_ID}_${TS}"

mkdir -p "$OUT_DIR/episodes" "$OUT_DIR/summaries" "$OUT_DIR/logs"

cd "$ROOT"

echo "[TRACER] record Phase-A runtime rollout v0"
echo "[TRACER] terrain:     $TERRAIN"
echo "[TRACER] policy_id:   $POLICY_ID"
echo "[TRACER] duration:    $DURATION_SEC"
echo "[TRACER] out_dir:     $OUT_DIR"
echo "[TRACER] episode_id:  $EPISODE_ID"

echo
echo "========== ROS2 topic check before recording =========="
set +u
source /opt/ros/humble/setup.bash
if [ -f "$ROOT/ros2_ws/install/setup.bash" ]; then
  export COLCON_TRACE="${COLCON_TRACE:-}"
  source "$ROOT/ros2_ws/install/setup.bash"
fi
set -u

ros2 topic info -v /tracer/mpc_reference || true
timeout 3 ros2 topic echo --once /tracer/mpc_reference || true
timeout 3 ros2 topic echo --once /tracer/phase_a_policy_status || true

echo
echo "========== start recorder =========="
export TRACER_EPISODE_ID="$EPISODE_ID"
export TRACER_TERRAIN="$TERRAIN"
export TRACER_POLICY_ID="$POLICY_ID"
export TRACER_ROLLOUT_DURATION_SEC="$DURATION_SEC"
export TRACER_ROLLOUT_OUT_DIR="$OUT_DIR"

timeout "$(python3 - <<PY
d=float("$DURATION_SEC")
print(max(10.0, d + 8.0))
PY
)" \
/usr/bin/python3 scripts/runtime/tracer_record_rollout_episode_v0.py \
  2>&1 | tee "$OUT_DIR/logs/${EPISODE_ID}_recorder.log"

echo
echo "========== outputs =========="
find "$OUT_DIR" -maxdepth 2 -type f | sort

echo
echo "[TRACER] out_dir=$OUT_DIR"
