#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

POLICY_JSON="${TRACER_FUSION_POLICY_JSON:-$PWD/configs/highlevel_policy/tracer_fusion_policy_v1.json}"
DURATION="${TRACER_FEASIBLE_DURATION:-10}"

echo "========== TRACER Feasible Closed-Loop GMS-On V0 =========="
echo "[TRACER] policy_json = $POLICY_JSON"
echo "[TRACER] duration    = $DURATION"
echo "[TRACER] meta_policy = ${TRACER_META_GAIT_POLICY_KIND:-udp}"
echo "[TRACER] gms_enable  = ${TRACER_ENABLE_GMS:-1}"
echo "[TRACER] ram_gate    = ${TRACER_GMS_USE_RAM_GATE:-0}"
echo

echo "========== ensure docker containers =========="
sudo docker start a1_unitree_gazebo_docker >/dev/null 2>&1 || true
sudo docker start a1_cpp_ctrl_docker >/dev/null 2>&1 || true
sudo docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | grep -E 'a1|unitree|gazebo|ctrl' || true
echo

run_one() {
  local terrain_key="$1"
  local world="$2"
  local tag="$3"

  echo
  echo "============================================================"
  echo "[TRACER] RUN feasible terrain=$terrain_key world=$world tag=$tag"
  echo "============================================================"

  TRACER_FUSION_POLICY_JSON="$POLICY_JSON" \
  TRACER_META_GAIT_POLICY_KIND="${TRACER_META_GAIT_POLICY_KIND:-udp}" \
  TRACER_ENABLE_GMS="${TRACER_ENABLE_GMS:-1}" \
  TRACER_GMS_USE_RAM_GATE="${TRACER_GMS_USE_RAM_GATE:-0}" \
  TRACER_ENABLE_RAM_MONITOR="${TRACER_ENABLE_RAM_MONITOR:-0}" \
  TRACER_ENABLE_RAM_GATE_MONITOR="${TRACER_ENABLE_RAM_GATE_MONITOR:-0}" \
  scripts/runtime/tracer_run_fusion_policy_sanity_once.sh \
    "$terrain_key" \
    "$world" \
    "$DURATION" \
    "$tag"
}

# 1) easiest baseline: flat feasible locomotion
run_one "flat_normal" \
        "earth" \
        "feasible_v0_flat_normal_gmson"

# 2) feasible rough terrain with normal-forward expectation
run_one "rough_mid" \
        "tracer_rough_mid" \
        "feasible_v0_rough_mid_gmson"

# 3) feasible slope/uphill probe-like behavior
run_one "slope_5deg" \
        "tracer_slope_5deg" \
        "feasible_v0_slope_5deg_gmson"

echo
echo "========== feasible closed-loop sweep done =========="
echo "[TRACER] latest sanity rows:"
tail -10 data/sanity_results/tracer_fusion_sanity_results.csv
