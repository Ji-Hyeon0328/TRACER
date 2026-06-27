#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

TERRAINS="${TRACER_GMS_ONLY_ROBUST_TERRAINS:-flat_normal rough_mid slope_5deg}"
REPEATS="${TRACER_GMS_ONLY_ROBUST_REPEATS:-3}"
DURATION="${TRACER_GMS_ONLY_ROBUST_DURATION_SEC:-30.0}"
SAMPLE_HZ="${TRACER_GMS_ONLY_ROBUST_SAMPLE_HZ:-5.0}"
POLICY_PREFIX="${TRACER_GMS_ONLY_ROBUST_POLICY_PREFIX:-theta_mlp_udp_v0_learned_stack_v3_gms_only_robust30}"

echo "[TRACER] learned stack v3 GMS-only robust eval"
echo "[TRACER] root:     $ROOT"
echo "[TRACER] terrains: $TERRAINS"
echo "[TRACER] repeats:  $REPEATS"
echo "[TRACER] duration: $DURATION"
echo "[TRACER] sample_hz: $SAMPLE_HZ"
echo "[TRACER] prefix:   $POLICY_PREFIX"
echo

for terrain in $TERRAINS; do
  echo
  echo "========== TRACER GMS-ONLY ROBUST EVAL: $terrain =========="

  TRACER_ROLLOUT_TERRAINS="$terrain" \
  TRACER_ROLLOUT_REPEATS="$REPEATS" \
  TRACER_ROLLOUT_DURATION_SEC="$DURATION" \
  TRACER_ROLLOUT_SAMPLE_HZ="$SAMPLE_HZ" \
  TRACER_POLICY_ID="${POLICY_PREFIX}_${terrain}" \
  TRACER_META_GAIT_POLICY_KIND=udp \
  TRACER_ENABLE_LEARNED_STACK_V3=1 \
  TRACER_DEPLOY_LEARNED_STACK_V3=1 \
  TRACER_LEARNED_STACK_V3_PORT="${TRACER_LEARNED_STACK_V3_PORT:-50430}" \
  TRACER_OBJECTIVE_SELECTOR_VERBOSE=0 \
  scripts/runtime/tracer_run_policy_rollout_dataset_v0.sh
done

echo
echo "[TRACER] learned stack v3 GMS-only robust eval finished"
