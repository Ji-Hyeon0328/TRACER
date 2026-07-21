#!/usr/bin/env bash
set -euo pipefail

REPEAT_N="${TRACER_PHASE_F16_REPEAT_N:-1}"

TAGS=(m060 m015 p015 p060)
RESET_YS=(-0.60 -0.15 0.15 0.60)

echo "[TRACER] Phase-F16 lateral sweep true metric batch v0"
echo "  REPEAT_N=${REPEAT_N}"
echo "  tags=${TAGS[*]}"
echo "  reset_y=${RESET_YS[*]}"

for rep in $(seq 1 "${REPEAT_N}"); do
  for idx in "${!TAGS[@]}"; do
    base_tag="${TAGS[$idx]}"
    reset_y="${RESET_YS[$idx]}"

    if [ "${REPEAT_N}" = "1" ]; then
      run_tag="${base_tag}"
    else
      run_tag="${base_tag}_r${rep}"
    fi

    echo
    echo "============================================================"
    echo "[TRACER] F16 run tag=${run_tag} reset_y=${reset_y} repeat=${rep}/${REPEAT_N}"
    echo "============================================================"

    TRACER_PHASE_F3_TAG="${run_tag}" \
    TRACER_RESET_Y_OFFSET="${reset_y}" \
    TRACER_PHASE_F3_F1_DURATION="${TRACER_PHASE_F3_F1_DURATION:-170}" \
    TRACER_PHASE_F3_F1_HZ="${TRACER_PHASE_F3_F1_HZ:-50}" \
    TRACER_PHASE_F3_TIMEOUT_S="${TRACER_PHASE_F3_TIMEOUT_S:-240}" \
    bash scripts/training/tracer_run_phase_f3_true_metric_rollout_with_f1_v0.sh
  done
done

echo
echo "[TRACER] F16 lateral sweep done"
