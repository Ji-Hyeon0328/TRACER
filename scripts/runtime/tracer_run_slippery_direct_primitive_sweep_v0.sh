#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

POLICY_JSON="$PWD/configs/highlevel_policy/tracer_fusion_policy_slippery_direct_primitive_sweep_v0.json"
DURATION="${TRACER_SWEEP_DURATION:-10}"

echo "========== TRACER slippery direct primitive sweep v0 =========="
echo "[TRACER] policy   = $POLICY_JSON"
echo "[TRACER] duration = $DURATION"
echo "[TRACER] meta     = ${TRACER_META_GAIT_POLICY_KIND:-udp}"
echo "[TRACER] gms      = ${TRACER_ENABLE_GMS:-0}"
echo

sudo docker start a1_unitree_gazebo_docker >/dev/null 2>&1 || true
sudo docker start a1_cpp_ctrl_docker >/dev/null 2>&1 || true

run_one() {
  local terrain_key="$1"
  local world="$2"
  local tag="$3"

  echo
  echo "============================================================"
  echo "[TRACER] terrain=$terrain_key"
  echo "[TRACER] world=$world"
  echo "[TRACER] tag=$tag"
  echo "============================================================"

  TRACER_FUSION_POLICY_JSON="$POLICY_JSON" \
  TRACER_META_GAIT_POLICY_KIND="${TRACER_META_GAIT_POLICY_KIND:-udp}" \
  TRACER_ENABLE_GMS="${TRACER_ENABLE_GMS:-0}" \
  scripts/runtime/tracer_run_fusion_policy_sanity_once.sh \
    "$terrain_key" \
    "$world" \
    "$DURATION" \
    "$tag"
}

# Slippery flat variants
run_one "slippery_mid_flat_crawl_02" \
        "tracer_slippery_mid_flat" \
        "direct_v0_slipflat_crawl_02"

run_one "slippery_mid_flat_crawl_04" \
        "tracer_slippery_mid_flat" \
        "direct_v0_slipflat_crawl_04"

run_one "slippery_mid_flat_crawl_06" \
        "tracer_slippery_mid_flat" \
        "direct_v0_slipflat_crawl_06"

run_one "slippery_mid_flat_tall_crawl_04" \
        "tracer_slippery_mid_flat" \
        "direct_v0_slipflat_tall_crawl_04"

run_one "slippery_mid_flat_active_hold_h335_c080" \
        "tracer_slippery_mid_flat" \
        "direct_v0_slipflat_active_hold_h335_c080"

# Slippery downslope variants
run_one "slippery_downslope_active_hold_h305_c055" \
        "tracer_slippery_downslope_5deg" \
        "direct_v0_slipdown_active_hold_h305_c055"

run_one "slippery_downslope_active_hold_h335_c080" \
        "tracer_slippery_downslope_5deg" \
        "direct_v0_slipdown_active_hold_h335_c080"

run_one "slippery_downslope_micro_backstep_02" \
        "tracer_slippery_downslope_5deg" \
        "direct_v0_slipdown_micro_backstep_02"

run_one "slippery_downslope_micro_backstep_04" \
        "tracer_slippery_downslope_5deg" \
        "direct_v0_slipdown_micro_backstep_04"

run_one "slippery_downslope_micro_backstep_06" \
        "tracer_slippery_downslope_5deg" \
        "direct_v0_slipdown_micro_backstep_06"

run_one "slippery_downslope_forward_crawl_02" \
        "tracer_slippery_downslope_5deg" \
        "direct_v0_slipdown_forward_crawl_02"

echo
echo "========== direct primitive sweep done =========="
