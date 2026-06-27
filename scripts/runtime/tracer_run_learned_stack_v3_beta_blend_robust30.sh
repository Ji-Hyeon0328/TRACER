#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

TERRAINS="${TRACER_ROBUST_TERRAINS:-flat_normal rough_mid slope_5deg}"
REPEATS="${TRACER_ROBUST_REPEATS:-3}"
DURATION="${TRACER_ROBUST_DURATION_SEC:-30.0}"
SAMPLE_HZ="${TRACER_ROBUST_SAMPLE_HZ:-5.0}"
ALPHA="${TRACER_LEARNED_STACK_V3_BETA_BLEND_ALPHA:-0.10}"

echo "[TRACER] learned stack v3 beta blend robust30"
echo "[TRACER] terrains:  ${TERRAINS}"
echo "[TRACER] repeats:   ${REPEATS}"
echo "[TRACER] duration:  ${DURATION}"
echo "[TRACER] sample_hz: ${SAMPLE_HZ}"
echo "[TRACER] beta alpha:${ALPHA}"

for terrain in ${TERRAINS}; do
  echo
  echo "========== BETA BLEND ROBUST30 ${terrain} =========="

  TRACER_ROLLOUT_TERRAINS="${terrain}" \
  TRACER_ROLLOUT_REPEATS="${REPEATS}" \
  TRACER_ROLLOUT_DURATION_SEC="${DURATION}" \
  TRACER_ROLLOUT_SAMPLE_HZ="${SAMPLE_HZ}" \
  TRACER_POLICY_ID="theta_mlp_udp_v0_learned_stack_v3_beta_blend_robust30_${terrain}" \
  TRACER_META_GAIT_POLICY_KIND=udp \
  TRACER_ENABLE_LEARNED_STACK_V3=1 \
  TRACER_DEPLOY_LEARNED_STACK_V3=1 \
  TRACER_DEPLOY_LEARNED_STACK_V3_BETA_BLEND=1 \
  TRACER_LEARNED_STACK_V3_BETA_BLEND_ALPHA="${ALPHA}" \
  TRACER_LEARNED_STACK_V3_PORT=50430 \
  TRACER_OBJECTIVE_SELECTOR_VERBOSE=0 \
  scripts/runtime/tracer_run_policy_rollout_dataset_v0.sh
done

echo
echo "[TRACER] beta blend robust30 collection finished"
