#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

WORLD="${TRACER_PHASE_B_WORLD:-earth}"
DURATION="${TRACER_PHASE_B_RECORD_DURATION:-30.0}"
SAMPLE_HZ="${TRACER_PHASE_B_SAMPLE_HZ:-20.0}"
THETA_MODEL="${TRACER_PHASE_B_THETA_MODEL:-}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${TRACER_PHASE_B_HC_SWEEP_RUN_ROOT:-$ROOT/artifacts/phase_b_theta_hc_sweep_${STAMP}}"
mkdir -p "$RUN_ROOT"

SUMMARY_CSV="$RUN_ROOT/hc_sweep_summary.csv"

echo "[TRACER] Phase-B theta height/clearance sweep"
echo "[TRACER] root:      $ROOT"
echo "[TRACER] world:     $WORLD"
echo "[TRACER] duration:  $DURATION"
echo "[TRACER] sample_hz: $SAMPLE_HZ"
echo "[TRACER] run_root:  $RUN_ROOT"

cat > "$RUN_ROOT/risk_low.json" <<'JSON'
{
  "ram_mismatch": 0.04,
  "ram_uncertainty": 0.04,
  "recent_max_yaw_rate": 0.08,
  "recent_slip_score": 0.02,
  "body_stability_score": 0.96,
  "yaw_p95_abs_rate": 0.06,
  "yaw_mean_abs_rate": 0.02,
  "yaw_saturation_fraction": 0.0
}
JSON

cat > "$RUN_ROOT/risk_moderate.json" <<'JSON'
{
  "ram_mismatch": 0.16,
  "ram_uncertainty": 0.17,
  "recent_max_yaw_rate": 0.20,
  "recent_slip_score": 0.12,
  "body_stability_score": 0.86,
  "yaw_p95_abs_rate": 0.18,
  "yaw_mean_abs_rate": 0.06,
  "yaw_saturation_fraction": 0.03
}
JSON

cat > "$RUN_ROOT/risk_high.json" <<'JSON'
{
  "ram_mismatch": 0.45,
  "ram_uncertainty": 0.45,
  "recent_max_yaw_rate": 0.30,
  "recent_slip_score": 0.35,
  "body_stability_score": 0.72,
  "yaw_p95_abs_rate": 0.29,
  "yaw_mean_abs_rate": 0.12,
  "yaw_saturation_fraction": 0.25
}
JSON

echo "case_name,risk_name,body_height,swing_clearance,reached,min_rel_dist,final_rel_dist,progress,odom_x_delta,max_vx,max_yaw,vx_far,vx_near,slow_distance,run_dir,summary_json" \
  > "$SUMMARY_CSV"

# Keep this first sweep intentionally small.
# Later we can expand per-terrain.
CONFIGS=(
  "low_base low 0.32 0.04"
  "low_low_clearance low 0.32 0.03"
  "low_high_clearance low 0.32 0.07"
  "low_high_body low 0.35 0.04"
  "moderate_base moderate 0.32 0.04"
  "moderate_high_clearance moderate 0.32 0.07"
  "moderate_high_body_clearance moderate 0.35 0.07"
  "high_base high 0.32 0.04"
  "high_low_body_high_clearance high 0.30 0.08"
)

idx=0

for CFG in "${CONFIGS[@]}"; do
  idx=$((idx + 1))
  read -r CASE_NAME RISK_NAME BODY_HEIGHT SWING_CLEARANCE <<< "$CFG"

  RISK_JSON="$RUN_ROOT/risk_${RISK_NAME}.json"
  CASE_DIR="$RUN_ROOT/case_$(printf "%03d" "$idx")_${CASE_NAME}"

  echo
  echo "============================================================"
  echo "[TRACER] HC sweep case $idx/${#CONFIGS[@]}: $CASE_NAME"
  echo "[TRACER] risk=$RISK_NAME body_height=$BODY_HEIGHT swing_clearance=$SWING_CLEARANCE"
  echo "============================================================"

  if [[ -n "$THETA_MODEL" ]]; then
    TRACER_PHASE_B_THETA_MODEL="$THETA_MODEL" \
    TRACER_PHASE_B_RISK_STATE_JSON="$RISK_JSON" \
    TRACER_PHASE_B_WORLD="$WORLD" \
    TRACER_PHASE_B_RECORD_DURATION="$DURATION" \
    TRACER_PHASE_B_SAMPLE_HZ="$SAMPLE_HZ" \
    TRACER_PHASE_B_BODY_HEIGHT="$BODY_HEIGHT" \
    TRACER_PHASE_B_SWING_CLEARANCE="$SWING_CLEARANCE" \
    TRACER_PHASE_B_RUN_DIR="$CASE_DIR" \
    "$ROOT/scripts/runtime/tracer_run_phase_b_theta_predicted_goal_episode_v0.sh" \
      2>&1 | tee "$CASE_DIR.log"
  else
    TRACER_PHASE_B_RISK_STATE_JSON="$RISK_JSON" \
    TRACER_PHASE_B_WORLD="$WORLD" \
    TRACER_PHASE_B_RECORD_DURATION="$DURATION" \
    TRACER_PHASE_B_SAMPLE_HZ="$SAMPLE_HZ" \
    TRACER_PHASE_B_BODY_HEIGHT="$BODY_HEIGHT" \
    TRACER_PHASE_B_SWING_CLEARANCE="$SWING_CLEARANCE" \
    TRACER_PHASE_B_RUN_DIR="$CASE_DIR" \
    "$ROOT/scripts/runtime/tracer_run_phase_b_theta_predicted_goal_episode_v0.sh" \
      2>&1 | tee "$CASE_DIR.log"
  fi

  SUMMARY_JSON="$CASE_DIR/episode/phase_b_summary.json"

  if [[ ! -s "$SUMMARY_JSON" ]]; then
    echo "[TRACER][WARN] missing summary: $SUMMARY_JSON"
    continue
  fi

  python3 - "$CASE_NAME" "$RISK_NAME" "$BODY_HEIGHT" "$SWING_CLEARANCE" "$CASE_DIR" "$SUMMARY_JSON" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys

case_name, risk_name, body_height, swing_clearance, run_dir, summary_json, summary_csv = sys.argv[1:]

with open(summary_json, "r") as f:
    s = json.load(f)

row = {
    "case_name": case_name,
    "risk_name": risk_name,
    "body_height": float(body_height),
    "swing_clearance": float(swing_clearance),
    "reached": s.get("reached_stop_distance"),
    "min_rel_dist": s.get("min_rel_dist"),
    "final_rel_dist": s.get("final_rel_dist"),
    "progress": s.get("progress_initial_minus_min"),
    "odom_x_delta": s.get("odom_x_delta"),
    "max_vx": s.get("max_mpc_vx"),
    "max_yaw": s.get("max_abs_mpc_yaw_rate"),
    "vx_far": s.get("vx_far"),
    "vx_near": s.get("vx_near"),
    "slow_distance": s.get("goal_slow_distance"),
    "run_dir": run_dir,
    "summary_json": summary_json,
}

with open(summary_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "case_name",
        "risk_name",
        "body_height",
        "swing_clearance",
        "reached",
        "min_rel_dist",
        "final_rel_dist",
        "progress",
        "odom_x_delta",
        "max_vx",
        "max_yaw",
        "vx_far",
        "vx_near",
        "slow_distance",
        "run_dir",
        "summary_json",
    ])
    writer.writerow(row)

print(json.dumps(row, indent=2, sort_keys=True))
PY

done

echo
echo "============================================================"
echo "[TRACER] HC sweep summary"
echo "============================================================"
column -s, -t < "$SUMMARY_CSV" || cat "$SUMMARY_CSV"

echo
echo "[TRACER] outputs:"
echo "  run_root: $RUN_ROOT"
echo "  summary:  $SUMMARY_CSV"
