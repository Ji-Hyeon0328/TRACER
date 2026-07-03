#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

WORLD="${TRACER_PHASE_B_WORLD:-tracer_sponge_firm_flat}"
DURATION="${TRACER_PHASE_B_RECORD_DURATION:-45.0}"
SAMPLE_HZ="${TRACER_PHASE_B_SAMPLE_HZ:-20.0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${TRACER_PHASE_B_SPONGE_PROBE_RUN_ROOT:-$ROOT/artifacts/phase_b_sponge_conservative_theta_probe_${STAMP}}"
mkdir -p "$RUN_ROOT"

SUMMARY_CSV="$RUN_ROOT/sponge_probe_summary.csv"

echo "[TRACER] sponge conservative theta probe"
echo "[TRACER] world:     $WORLD"
echo "[TRACER] duration:  $DURATION"
echo "[TRACER] sample_hz: $SAMPLE_HZ"
echo "[TRACER] run_root:  $RUN_ROOT"

echo "case_name,world,vx_far,vx_near,slow_distance,body_height,swing_clearance,reached,min_rel_dist,final_rel_dist,progress,odom_x_delta,max_yaw,run_dir,summary_json" \
  > "$SUMMARY_CSV"

CONFIGS=(
  "sponge_slow_h034_c008 0.04 0.025 0.30 0.34 0.08"
  "sponge_slow_h036_c010 0.04 0.025 0.30 0.36 0.10"
  "sponge_slow_h038_c012 0.04 0.025 0.30 0.38 0.12"
  "sponge_mid_h034_c008 0.06 0.035 0.30 0.34 0.08"
  "sponge_mid_h036_c010 0.06 0.035 0.30 0.36 0.10"
  "sponge_mid_h038_c012 0.06 0.035 0.30 0.38 0.12"
  "sponge_probe_h036_c012 0.08 0.04 0.28 0.36 0.12"
  "sponge_probe_h038_c014 0.08 0.04 0.28 0.38 0.14"
)

idx=0

for CFG in "${CONFIGS[@]}"; do
  idx=$((idx + 1))
  read -r CASE VX_FAR VX_NEAR SLOW BODY_H CLR <<< "$CFG"

  CASE_DIR="$RUN_ROOT/case_$(printf "%03d" "$idx")_${CASE}"

  echo
  echo "============================================================"
  echo "[TRACER] sponge probe $idx/${#CONFIGS[@]} case=$CASE"
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
  else
    echo "[TRACER][WARN] missing summary: $SUMMARY_JSON"
  fi
done

echo
echo "============================================================"
echo "[TRACER] sponge probe summary"
echo "============================================================"
column -s, -t < "$SUMMARY_CSV" || cat "$SUMMARY_CSV"

echo
echo "[TRACER] outputs:"
echo "  run_root: $RUN_ROOT"
echo "  summary:  $SUMMARY_CSV"
