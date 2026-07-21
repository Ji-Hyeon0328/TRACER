#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "[TRACER][G4] Anchor sweep true-metric batch"
echo "[TRACER][G4] cwd=$(pwd)"
echo "[TRACER][G4] start=$(date)"

run_one() {
  local tag="$1"
  local reset_y="$2"

  echo
  echo "============================================================"
  echo "[TRACER][G4] tag=${tag} reset_y=${reset_y}"
  echo "============================================================"

  TRACER_PHASE_F3_TAG="${tag}" \
  TRACER_RESET_Y_OFFSET="${reset_y}" \
    bash scripts/training/tracer_run_phase_f3_true_metric_rollout_with_f1_v0.sh
}

# Positive-side anchors around the problematic p060 condition.
run_one "p045"     "0.45"
run_one "p045_r1"  "0.45"
run_one "p045_r2"  "0.45"

run_one "p060_r3"  "0.60"
run_one "p060_r4"  "0.60"
run_one "p060_r5"  "0.60"

run_one "p075"     "0.75"
run_one "p075_r1"  "0.75"
run_one "p075_r2"  "0.75"

# Negative-side anchors for symmetry / extrapolation check.
run_one "m045"     "-0.45"
run_one "m045_r1"  "-0.45"
run_one "m045_r2"  "-0.45"

run_one "m075"     "-0.75"
run_one "m075_r1"  "-0.75"
run_one "m075_r2"  "-0.75"

echo
echo "[TRACER][G4] done=$(date)"
