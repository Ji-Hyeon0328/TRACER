#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"

WORLD="${TRACER_D4_REPEAT_WORLD:-tracer_mixed_stress_course_v5_lowfric_from_solid}"
GOAL_X="${TRACER_D4_REPEAT_GOAL_X:-8.0}"
LATERAL_BOUND="${TRACER_D4_REPEAT_LATERAL_BOUND:-2.0}"
N="${TRACER_D4_TABLE_BANK_N:-3}"
TIMEOUT_S="${TRACER_D4_REPEAT_TIMEOUT_S:-260}"
HOLD_OBS_S="${TRACER_D4_HOLD_OBS_S:-60}"
HOLD_VX="${TRACER_PHASE_D4_HOLD_VX:-0.025}"

TS="$(date +%Y%m%d_%H%M%S)"
BANK_MANIFEST="$ROOT/reports/phase_d4_context_table_bank_${TS}_summaries.tsv"
CAND_FILE="$ROOT/reports/phase_d4_context_table_bank_${TS}_candidates.tsv"

mkdir -p "$ROOT/reports/phase_d4"

cat > "$CAND_FILE" <<'EOF'
name	action_table
baseline_hold025	flat:0.210,0.320,0.045;start_flat:0.210,0.320,0.045;upslope:0.210,0.320,0.045;rough:0.210,0.320,0.045;downslope:0.2025,0.320,0.045;goal_flat:0.2025,0.320,0.045;unknown:0.2025,0.320,0.045
conservative_down	flat:0.210,0.320,0.045;start_flat:0.210,0.320,0.045;upslope:0.210,0.320,0.045;rough:0.210,0.320,0.045;downslope:0.1900,0.320,0.045;goal_flat:0.1900,0.320,0.045;unknown:0.1900,0.320,0.045
rough_clearance	flat:0.210,0.320,0.045;start_flat:0.210,0.320,0.045;upslope:0.210,0.320,0.045;rough:0.2050,0.320,0.055;downslope:0.2025,0.320,0.045;goal_flat:0.2025,0.320,0.045;unknown:0.2025,0.320,0.045
smooth_all	flat:0.205,0.320,0.045;start_flat:0.205,0.320,0.045;upslope:0.205,0.320,0.045;rough:0.205,0.320,0.050;downslope:0.1950,0.320,0.045;goal_flat:0.1950,0.320,0.045;unknown:0.1950,0.320,0.045
aggressive_flat_rough	flat:0.215,0.320,0.045;start_flat:0.215,0.320,0.045;upslope:0.210,0.320,0.045;rough:0.215,0.320,0.050;downslope:0.2025,0.320,0.045;goal_flat:0.2025,0.320,0.045;unknown:0.2025,0.320,0.045
EOF

echo -e "name\taction_table\thold_vx\tmanifest\tsummary_json\tsummary_md" > "$BANK_MANIFEST"

echo "[TRACER] Phase-D4 context table bank v0"
echo "  WORLD=$WORLD"
echo "  GOAL_X=$GOAL_X"
echo "  LATERAL_BOUND=$LATERAL_BOUND"
echo "  N=$N"
echo "  HOLD_VX=$HOLD_VX"
echo "  CAND_FILE=$CAND_FILE"
echo "  BANK_MANIFEST=$BANK_MANIFEST"

cd "$ROOT"

tail -n +2 "$CAND_FILE" | while IFS=$'\t' read -r name action_table; do
  echo
  echo "============================================================"
  echo "[TRACER] context-table candidate=$name"
  echo "============================================================"
  echo "$action_table"

  TRACER_PHASE_D4_CONTEXT_ACTION_TABLE="$action_table" \
  TRACER_PHASE_D4_HOLD_VX="$HOLD_VX" \
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
  OUT_PREFIX="reports/phase_d4/context_table_${name}_$(basename "$MANIFEST" _manifest.tsv)"

  python3 scripts/training/tracer_eval_phase_d4_context_meta_logs_v0.py \
    --manifest "$MANIFEST" \
    --out-prefix "$OUT_PREFIX"

  echo -e "${name}\t${action_table}\t${HOLD_VX}\t${MANIFEST}\t${OUT_PREFIX}_summary_v0.json\t${OUT_PREFIX}_summary_v0.md" >> "$BANK_MANIFEST"
done

echo
echo "[TRACER] context table bank finished"
echo "BANK_MANIFEST: $BANK_MANIFEST"
