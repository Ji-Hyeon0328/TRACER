#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
HOLD_VX_LIST="${TRACER_D4_HOLD_VX_LIST:-0.000 0.015 0.025 0.035 0.050}"
N="${TRACER_D4_HOLD_BANK_N:-3}"

WORLD="${TRACER_D4_REPEAT_WORLD:-tracer_mixed_stress_course_v5_lowfric_from_solid}"
GOAL_X="${TRACER_D4_REPEAT_GOAL_X:-8.0}"
LATERAL_BOUND="${TRACER_D4_REPEAT_LATERAL_BOUND:-2.0}"
TIMEOUT_S="${TRACER_D4_REPEAT_TIMEOUT_S:-260}"
HOLD_OBS_S="${TRACER_D4_HOLD_OBS_S:-60}"

TS="$(date +%Y%m%d_%H%M%S)"
BANK_MANIFEST="$ROOT/reports/phase_d4_hold_vx_bank_${TS}_summaries.tsv"

mkdir -p "$ROOT/reports/phase_d4"

echo -e "hold_vx\tmanifest\tsummary_json\tsummary_md" > "$BANK_MANIFEST"

echo "[TRACER] Phase-D4 hold_vx bank v0"
echo "  HOLD_VX_LIST=$HOLD_VX_LIST"
echo "  N=$N"
echo "  WORLD=$WORLD"
echo "  GOAL_X=$GOAL_X"
echo "  BANK_MANIFEST=$BANK_MANIFEST"

cd "$ROOT"

for hv in $HOLD_VX_LIST; do
  tag="$(printf '%s' "$hv" | sed 's/\./p/g')"
  echo
  echo "============================================================"
  echo "[TRACER] hold_vx=$hv"
  echo "============================================================"

  TRACER_PHASE_D4_HOLD_VX="$hv" \
  TRACER_D4_REPEAT_WORLD="$WORLD" \
  TRACER_D4_REPEAT_GOAL_X="$GOAL_X" \
  TRACER_D4_REPEAT_LATERAL_BOUND="$LATERAL_BOUND" \
  TRACER_D4_REPEAT_N="$N" \
  TRACER_D4_REPEAT_TIMEOUT_S="$TIMEOUT_S" \
  TRACER_D4_HOLD_OBS_S="$HOLD_OBS_S" \
  TRACER_D4_REPEAT_OPEN_GUI=0 \
  TRACER_PHASE_D4_STOP_MARGIN="${TRACER_PHASE_D4_STOP_MARGIN:--0.05}" \
  TRACER_PHASE_D4_YAW_SIGN="${TRACER_PHASE_D4_YAW_SIGN:-0.0}" \
  TRACER_PHASE_D4_YAW_K="${TRACER_PHASE_D4_YAW_K:-0.0}" \
  TRACER_PHASE_D4_YAW_MAX="${TRACER_PHASE_D4_YAW_MAX:-0.0}" \
  bash scripts/runtime/tracer_run_phase_d4_context_meta_repeat_v0.sh

  MANIFEST="$(ls -t reports/phase_d4_context_meta_repeat_*_manifest.tsv | head -1)"
  OUT_PREFIX="reports/phase_d4/hold_vx_${tag}_$(basename "$MANIFEST" _manifest.tsv)"

  python3 scripts/training/tracer_eval_phase_d4_context_meta_logs_v0.py \
    --manifest "$MANIFEST" \
    --out-prefix "$OUT_PREFIX"

  echo -e "${hv}\t${MANIFEST}\t${OUT_PREFIX}_summary_v0.json\t${OUT_PREFIX}_summary_v0.md" >> "$BANK_MANIFEST"
done

echo
echo "[TRACER] hold_vx bank finished"
echo "BANK_MANIFEST: $BANK_MANIFEST"
