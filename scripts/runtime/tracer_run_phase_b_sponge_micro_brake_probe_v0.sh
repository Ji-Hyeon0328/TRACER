#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

WORLD="${TRACER_PHASE_B_WORLD:-tracer_sponge_firm_flat}"
DURATION="${TRACER_PHASE_B_RECORD_DURATION:-45.0}"
SAMPLE_HZ="${TRACER_PHASE_B_SAMPLE_HZ:-20.0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${TRACER_PHASE_B_SPONGE_MICRO_RUN_ROOT:-$ROOT/artifacts/phase_b_sponge_micro_brake_probe_${STAMP}}"
mkdir -p "$RUN_ROOT"

SUMMARY_CSV="$RUN_ROOT/sponge_micro_brake_summary.csv"

echo "case_name,world,vx_far,vx_near,slow_distance,body_height,swing_clearance,reached,min_rel_dist,final_rel_dist,progress,odom_x_delta,max_yaw,run_dir,summary_json" \
  > "$SUMMARY_CSV"

CONFIGS=(
  "micro_v020_n010_s040_h034_c008 0.020 0.010 0.40 0.34 0.08"
  "micro_v025_n010_s040_h034_c008 0.025 0.010 0.40 0.34 0.08"
  "micro_v030_n015_s040_h034_c008 0.030 0.015 0.40 0.34 0.08"
  "micro_v035_n015_s040_h034_c008 0.035 0.015 0.40 0.34 0.08"
  "micro_v040_n015_s045_h034_c008 0.040 0.015 0.45 0.34 0.08"
  "micro_v030_n010_s045_h032_c006 0.030 0.010 0.45 0.32 0.06"
  "micro_v030_n010_s045_h034_c006 0.030 0.010 0.45 0.34 0.06"
  "micro_v030_n010_s045_h034_c010 0.030 0.010 0.45 0.34 0.10"
)

idx=0

for CFG in "${CONFIGS[@]}"; do
  idx=$((idx + 1))
  read -r CASE VX_FAR VX_NEAR SLOW BODY_H CLR <<< "$CFG"

  CASE_DIR="$RUN_ROOT/case_$(printf "%03d" "$idx")_${CASE}"

  echo
  echo "============================================================"
  echo "[TRACER] sponge micro-brake $idx/${#CONFIGS[@]} case=$CASE"
  echo "[TRACER] vx=$VX_FAR near=$VX_NEAR slow=$SLOW h=$BODY_H clr=$CLR"
  echo "============================================================"

  TRACER_PHASE_B_WORLD="$WORLD" \
  TRACER_PHASE_B_RUN_DIR="$CASE_DIR" \
  TRACER_PHASE_B_GOAL_DISTANCE_AHEAD=0.5 \
  TRACER_PHASE_B_GOAL_STOP_DISTANCE=0.15 \
  TRACER_PHASE_B_GOAL_SLOW_DISTANCE="$SLOW" \
  TRACER_PHASE_B_VX_FAR="$VX_FAR" \
  TRACER_PHASE_B_VX_NEAR="$VX_NEAR" \
  TRACER_PHASE_B_BODY_HEIGHT="$BODY_H" \
  TRACER_PHASE_B_SWING_CLEARANCE="$CLR" \
  TRACER_PHASE_B_RECORD_DURATION="$DURATION" \
  TRACER_PHASE_B_SAMPLE_HZ="$SAMPLE_HZ" \
  "$ROOT/scripts/runtime/tracer_run_phase_b_goal_episode_v0.sh" \
    2>&1 | tee "$CASE_DIR.log"

  SUMMARY_JSON="$CASE_DIR/phase_b_summary.json"

  if [[ -s "$SUMMARY_JSON" ]]; then
    python3 - "$CASE" "$WORLD" "$CASE_DIR" "$SUMMARY_JSON" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys

case, world, run_dir, summary_json, summary_csv = sys.argv[1:]
s = json.load(open(summary_json))

row = {
    "case_name": case,
    "world": world,
    "vx_far": s.get("vx_far"),
    "vx_near": s.get("vx_near"),
    "slow_distance": s.get("goal_slow_distance"),
    "body_height": s.get("body_height"),
    "swing_clearance": s.get("swing_clearance"),
    "reached": s.get("reached_stop_distance"),
    "min_rel_dist": s.get("min_rel_dist"),
    "final_rel_dist": s.get("final_rel_dist"),
    "progress": s.get("progress_initial_minus_min"),
    "odom_x_delta": s.get("odom_x_delta"),
    "max_yaw": s.get("max_abs_mpc_yaw_rate"),
    "run_dir": run_dir,
    "summary_json": summary_json,
}

with open(summary_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)

print(json.dumps(row, indent=2, sort_keys=True))
PY
  fi
done

echo
column -s, -t < "$SUMMARY_CSV" || cat "$SUMMARY_CSV"
