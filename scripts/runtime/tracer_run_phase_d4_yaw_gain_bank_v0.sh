#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"

POLICY_JSON="${TRACER_PHASE_D4_POLICY_JSON:-models/phase_d4/d4_context_meta_policy_rough_clearance_holdvx0025_v0.json}"
YAW_GAIN_LIST="${TRACER_D4_YAW_GAIN_LIST:-0.000 0.050 -0.050 0.100 -0.100}"
N="${TRACER_D4_YAW_BANK_N:-3}"

WORLD="${TRACER_D4_REPEAT_WORLD:-tracer_mixed_stress_course_v5_lowfric_from_solid}"
GOAL_X="${TRACER_D4_REPEAT_GOAL_X:-8.0}"
LATERAL_BOUND="${TRACER_D4_REPEAT_LATERAL_BOUND:-2.0}"
TIMEOUT_S="${TRACER_D4_REPEAT_TIMEOUT_S:-260}"
HOLD_OBS_S="${TRACER_D4_HOLD_OBS_S:-60}"
HOLD_VX="${TRACER_PHASE_D4_HOLD_VX:-0.025}"
YAW_MAX="${TRACER_PHASE_D4_YAW_MAX:-0.25}"

TS="$(date +%Y%m%d_%H%M%S)"
BANK_MANIFEST="$ROOT/reports/phase_d4_yaw_gain_bank_${TS}_summaries.tsv"

mkdir -p "$ROOT/reports/phase_d4"

echo -e "yaw_gain\tyaw_sign\tyaw_k\tyaw_max\tpolicy_json\tmanifest\tsummary_json\tsummary_md" > "$BANK_MANIFEST"

echo "[TRACER] Phase-D4 yaw gain bank v0"
echo "  POLICY_JSON=$POLICY_JSON"
echo "  YAW_GAIN_LIST=$YAW_GAIN_LIST"
echo "  N=$N"
echo "  WORLD=$WORLD"
echo "  GOAL_X=$GOAL_X"
echo "  HOLD_VX=$HOLD_VX"
echo "  YAW_MAX=$YAW_MAX"
echo "  BANK_MANIFEST=$BANK_MANIFEST"

cd "$ROOT"

for yg in $YAW_GAIN_LIST; do
  sign="$(python3 - "$yg" <<'PY'
import sys
x = float(sys.argv[1])
print("1.0" if x >= 0 else "-1.0")
PY
)"
  k="$(python3 - "$yg" <<'PY'
import sys
print(abs(float(sys.argv[1])))
PY
)"
  tag="$(printf '%s' "$yg" | sed 's/-/m/g; s/\./p/g')"

  echo
  echo "============================================================"
  echo "[TRACER] yaw_gain=$yg sign=$sign k=$k"
  echo "============================================================"

  TRACER_PHASE_D4_POLICY_JSON="$POLICY_JSON" \
  TRACER_PHASE_D4_HOLD_VX="$HOLD_VX" \
  TRACER_D4_REPEAT_WORLD="$WORLD" \
  TRACER_D4_REPEAT_GOAL_X="$GOAL_X" \
  TRACER_D4_REPEAT_LATERAL_BOUND="$LATERAL_BOUND" \
  TRACER_D4_REPEAT_N="$N" \
  TRACER_D4_REPEAT_TIMEOUT_S="$TIMEOUT_S" \
  TRACER_D4_HOLD_OBS_S="$HOLD_OBS_S" \
  TRACER_D4_REPEAT_OPEN_GUI=0 \
  TRACER_PHASE_D4_STOP_MARGIN="${TRACER_PHASE_D4_STOP_MARGIN:--0.05}" \
  TRACER_PHASE_D4_YAW_SIGN="$sign" \
  TRACER_PHASE_D4_YAW_K="$k" \
  TRACER_PHASE_D4_YAW_MAX="$YAW_MAX" \
  bash scripts/runtime/tracer_run_phase_d4_context_meta_repeat_v0.sh

  MANIFEST="$(ls -t reports/phase_d4_context_meta_repeat_*_manifest.tsv | head -1)"
  OUT_PREFIX="reports/phase_d4/yaw_gain_${tag}_$(basename "$MANIFEST" _manifest.tsv)"

  python3 scripts/training/tracer_eval_phase_d4_context_meta_logs_v0.py \
    --manifest "$MANIFEST" \
    --out-prefix "$OUT_PREFIX"

  echo -e "${yg}\t${sign}\t${k}\t${YAW_MAX}\t${POLICY_JSON}\t${MANIFEST}\t${OUT_PREFIX}_summary_v0.json\t${OUT_PREFIX}_summary_v0.md" >> "$BANK_MANIFEST"
done

echo
echo "[TRACER] yaw gain bank finished"
echo "BANK_MANIFEST: $BANK_MANIFEST"
