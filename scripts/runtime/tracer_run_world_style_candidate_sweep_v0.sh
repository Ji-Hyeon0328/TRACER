#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"

TERRAINS="${TRACER_SWEEP_TERRAINS:-flat_normal rough_mid slope_5deg}"
STYLES="${TRACER_SWEEP_STYLES:-fast cautious high_clearance}"
REPEATS="${TRACER_SWEEP_REPEATS:-2}"
DURATION="${TRACER_SWEEP_DURATION_SEC:-20.0}"
SAMPLE_HZ="${TRACER_SWEEP_SAMPLE_HZ:-5.0}"
POLICY_ID_PREFIX="${TRACER_SWEEP_POLICY_ID_PREFIX:-world_style_sweep_v0}"

cd "$ROOT"

echo "[TRACER] world style candidate sweep"
echo "[TRACER] terrains: $TERRAINS"
echo "[TRACER] styles:   $STYLES"
echo "[TRACER] repeats:  $REPEATS"
echo "[TRACER] duration: $DURATION"

for terrain in $TERRAINS; do
  for style in $STYLES; do
    echo
    echo "========== STYLE SWEEP terrain=$terrain style=$style =========="

    # Override style through env consumed by fusion policy node if supported.
    # If the current policy node ignores this, the next diagnostic will reveal it.
    TRACER_FORCE_STYLE="$style" \
    TRACER_GAZEBO_RESTART_CONTAINER="${TRACER_GAZEBO_RESTART_CONTAINER:-1}" \
    TRACER_ROLLOUT_TERRAINS="$terrain" \
    TRACER_ROLLOUT_REPEATS="$REPEATS" \
    TRACER_ROLLOUT_DURATION_SEC="$DURATION" \
    TRACER_ROLLOUT_SAMPLE_HZ="$SAMPLE_HZ" \
    TRACER_POLICY_ID="${POLICY_ID_PREFIX}_${terrain}_${style}" \
    scripts/runtime/tracer_run_policy_rollout_dataset_worlds_v0.sh
  done
done

echo
echo "[TRACER] world style candidate sweep finished"
