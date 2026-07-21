#!/usr/bin/env bash
set -euo pipefail

REPEAT_N="${TRACER_PHASE_F11_REPEAT_N:-2}"

echo "[TRACER] Phase-F11 true metric repeat batch v0"
echo "  REPEAT_N=${REPEAT_N}"

run_one() {
  local tag="$1"
  local reset_y="$2"
  local i="$3"

  echo
  echo "============================================================"
  echo "[TRACER] F11 run tag=${tag} reset_y=${reset_y} repeat=${i}/${REPEAT_N}"
  echo "============================================================"

  TRACER_PHASE_F3_TAG="${tag}_r${i}" \
  TRACER_RESET_Y_OFFSET="${reset_y}" \
  TRACER_PHASE_F3_F1_DURATION="${TRACER_PHASE_F3_F1_DURATION:-170}" \
  TRACER_PHASE_F3_F1_HZ="${TRACER_PHASE_F3_F1_HZ:-50}" \
  TRACER_PHASE_F3_TIMEOUT_S="${TRACER_PHASE_F3_TIMEOUT_S:-240}" \
  bash scripts/training/tracer_run_phase_f3_true_metric_rollout_with_f1_v0.sh
}

for i in $(seq 1 "${REPEAT_N}"); do
  run_one clean 0.00 "${i}"
  run_one m030 -0.30 "${i}"
  run_one p030 0.30 "${i}"
done

echo
echo "[TRACER] F11 repeat batch done"
