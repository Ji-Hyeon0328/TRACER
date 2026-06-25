#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

POLICY_JSON="$PWD/configs/highlevel_policy/tracer_fusion_policy_slippery_direct_primitive_sweep_v0.json"
BASE_POLICY_JSON="$PWD/configs/highlevel_policy/tracer_fusion_policy_v1.json"
DURATION="${TRACER_SWEEP_DURATION:-10}"
REPEATS="${TRACER_REPEAT_N:-3}"

echo "========== TRACER slippery active-hold repeat v0 =========="
echo "[TRACER] direct policy = $POLICY_JSON"
echo "[TRACER] base policy   = $BASE_POLICY_JSON"
echo "[TRACER] duration      = $DURATION"
echo "[TRACER] repeats       = $REPEATS"
echo "[TRACER] meta          = ${TRACER_META_GAIT_POLICY_KIND:-udp}"
echo "[TRACER] gms           = ${TRACER_ENABLE_GMS:-0}"
echo

sudo docker start a1_unitree_gazebo_docker >/dev/null 2>&1 || true
sudo docker start a1_cpp_ctrl_docker >/dev/null 2>&1 || true

run_one() {
  local policy_json="$1"
  local terrain_key="$2"
  local world="$3"
  local tag="$4"

  echo
  echo "============================================================"
  echo "[TRACER] policy=$policy_json"
  echo "[TRACER] terrain=$terrain_key"
  echo "[TRACER] world=$world"
  echo "[TRACER] tag=$tag"
  echo "============================================================"

  TRACER_FUSION_POLICY_JSON="$policy_json" \
  TRACER_META_GAIT_POLICY_KIND="${TRACER_META_GAIT_POLICY_KIND:-udp}" \
  TRACER_ENABLE_GMS="${TRACER_ENABLE_GMS:-0}" \
  scripts/runtime/tracer_run_fusion_policy_sanity_once.sh \
    "$terrain_key" \
    "$world" \
    "$DURATION" \
    "$tag"
}

for i in $(seq 1 "$REPEATS"); do
  run_one "$BASE_POLICY_JSON" \
    "slippery_mid_flat" \
    "tracer_slippery_mid_flat" \
    "repeat_v0_base_slipflat_crawl06_r${i}"

  run_one "$POLICY_JSON" \
    "slippery_mid_flat_active_hold_h335_c080" \
    "tracer_slippery_mid_flat" \
    "repeat_v0_direct_slipflat_active_hold_h335_c080_r${i}"

  run_one "$BASE_POLICY_JSON" \
    "slippery_downslope_5deg_forward" \
    "tracer_slippery_downslope_5deg" \
    "repeat_v0_base_slipdown_safe_stop_r${i}"

  run_one "$POLICY_JSON" \
    "slippery_downslope_active_hold_h335_c080" \
    "tracer_slippery_downslope_5deg" \
    "repeat_v0_direct_slipdown_active_hold_h335_c080_r${i}"
done

echo
echo "========== repeat done =========="
