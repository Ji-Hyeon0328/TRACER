#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

THETA_MODEL="${TRACER_PHASE_B_THETA_MODEL:-}"
if [[ -z "$THETA_MODEL" ]]; then
  THETA_MODEL="$(ls -td artifacts/phase_b_theta_regressor_v2_* 2>/dev/null | head -1)/phase_b_theta_regressor_v2.json"
fi

if [[ ! -s "$THETA_MODEL" ]]; then
  THETA_MODEL="$(ls -td artifacts/phase_b_theta_regressor_v1_* 2>/dev/null | head -1)/phase_b_theta_regressor_v1.json"
fi

if [[ ! -s "$THETA_MODEL" ]]; then
  THETA_MODEL="$(ls -td artifacts/phase_b_theta_regressor_v0_* 2>/dev/null | head -1)/phase_b_theta_regressor_v0.json"
fi

if [[ ! -s "$THETA_MODEL" ]]; then
  echo "[TRACER][ERROR] theta model not found: $THETA_MODEL"
  exit 1
fi

WORLD="${TRACER_PHASE_B_WORLD:-earth}"
NUM_EPISODES="${TRACER_PHASE_B_ADAPTIVE_EPISODES:-5}"
DURATION="${TRACER_PHASE_B_RECORD_DURATION:-30.0}"
SAMPLE_HZ="${TRACER_PHASE_B_SAMPLE_HZ:-20.0}"
EMA_ALPHA="${TRACER_PHASE_B_RISK_EMA_ALPHA:-0.45}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${TRACER_PHASE_B_ADAPTIVE_RUN_ROOT:-$ROOT/artifacts/phase_b_adaptive_theta_episodes_${STAMP}}"
mkdir -p "$RUN_ROOT"

SUMMARY_CSV="$RUN_ROOT/adaptive_theta_summary.csv"

echo "[TRACER] Phase-B adaptive continuous theta episodes"
echo "[TRACER] root:          $ROOT"
echo "[TRACER] theta_model:   $THETA_MODEL"
echo "[TRACER] world:         $WORLD"
echo "[TRACER] num_episodes:  $NUM_EPISODES"
echo "[TRACER] duration:      $DURATION"
echo "[TRACER] ema_alpha:     $EMA_ALPHA"
echo "[TRACER] run_root:      $RUN_ROOT"

echo "episode,theta_guard,reached,min_rel_dist,final_rel_dist,progress,odom_x_delta,max_vx,max_yaw,vx_far,vx_near,slow_distance,risk_mismatch,risk_uncertainty,risk_yaw,risk_slip,risk_stability,risk_json,summary_json" \
  > "$SUMMARY_CSV"

RISK_JSON="$RUN_ROOT/risk_state_000.json"

if [[ -n "${TRACER_PHASE_B_INITIAL_RISK_STATE_JSON:-}" ]]; then
  cp "${TRACER_PHASE_B_INITIAL_RISK_STATE_JSON}" "$RISK_JSON"
else
  python3 "$ROOT/scripts/runtime/tracer_make_phase_b_risk_state_from_summary_v2.py" \
    --ema-alpha "$EMA_ALPHA" \
    --out-json "$RISK_JSON" \
    > "$RUN_ROOT/risk_state_000_stdout.json"
fi

for EP in $(seq 1 "$NUM_EPISODES"); do
  EP_PAD="$(printf "%03d" "$EP")"
  EP_DIR="$RUN_ROOT/episode_${EP_PAD}"

  echo
  echo "============================================================"
  echo "[TRACER] adaptive theta episode $EP/$NUM_EPISODES"
  echo "[TRACER] risk_json: $RISK_JSON"
  echo "[TRACER] ep_dir:    $EP_DIR"
  echo "============================================================"

  TRACER_PHASE_B_THETA_MODEL="$THETA_MODEL" \
  TRACER_PHASE_B_RISK_STATE_JSON="$RISK_JSON" \
  TRACER_PHASE_B_WORLD="$WORLD" \
  TRACER_PHASE_B_RECORD_DURATION="$DURATION" \
  TRACER_PHASE_B_SAMPLE_HZ="$SAMPLE_HZ" \
  TRACER_PHASE_B_RUN_DIR="$EP_DIR" \
  "$ROOT/scripts/runtime/tracer_run_phase_b_theta_predicted_goal_episode_v0.sh" \
    2>&1 | tee "$EP_DIR.log"

  SUMMARY_JSON="$EP_DIR/episode/phase_b_summary.json"

  if [[ ! -s "$SUMMARY_JSON" ]]; then
    echo "[TRACER][ERROR] missing summary: $SUMMARY_JSON"
    exit 1
  fi

  python3 - "$EP" "$RISK_JSON" "$SUMMARY_JSON" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys

ep = int(sys.argv[1])
risk_path = sys.argv[2]
summary_path = sys.argv[3]
csv_path = sys.argv[4]

with open(risk_path, "r") as f:
    risk = json.load(f)

with open(summary_path, "r") as f:
    s = json.load(f)

guards = s.get("theta_guard_reasons", [])
if isinstance(guards, list):
    guard_text = "; ".join(str(x) for x in guards)
else:
    guard_text = str(guards)

row = {
    "episode": ep,
    "theta_guard": guard_text,
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
    "risk_mismatch": risk.get("ram_mismatch"),
    "risk_uncertainty": risk.get("ram_uncertainty"),
    "risk_yaw": risk.get("recent_max_yaw_rate"),
    "risk_slip": risk.get("recent_slip_score"),
    "risk_stability": risk.get("body_stability_score"),
    "risk_json": risk_path,
    "summary_json": summary_path,
}

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "episode",
        "theta_guard",
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
        "risk_mismatch",
        "risk_uncertainty",
        "risk_yaw",
        "risk_slip",
        "risk_stability",
        "risk_json",
        "summary_json",
    ])
    writer.writerow(row)

print(json.dumps(row, indent=2, sort_keys=True))
PY

  NEXT_EP="$((EP + 1))"
  NEXT_RISK_JSON="$RUN_ROOT/risk_state_$(printf "%03d" "$NEXT_EP").json"

  python3 "$ROOT/scripts/runtime/tracer_make_phase_b_risk_state_from_summary_v2.py" \
    --summary-json "$SUMMARY_JSON" \
    --prev-risk-json "$RISK_JSON" \
    --ema-alpha "$EMA_ALPHA" \
    --out-json "$NEXT_RISK_JSON" \
    > "$RUN_ROOT/risk_state_$(printf "%03d" "$NEXT_EP")_stdout.json"

  echo "[TRACER] next risk state:"
  python3 - "$NEXT_RISK_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    r = json.load(f)

print(json.dumps({
    "ram_mismatch": r.get("ram_mismatch"),
    "ram_uncertainty": r.get("ram_uncertainty"),
    "recent_max_yaw_rate": r.get("recent_max_yaw_rate"),
    "recent_slip_score": r.get("recent_slip_score"),
    "body_stability_score": r.get("body_stability_score"),
    "raw_max_abs_mpc_yaw_rate": r.get("raw_max_abs_mpc_yaw_rate"),
    "yaw_p95_abs_rate": r.get("yaw_p95_abs_rate"),
    "yaw_mean_abs_rate": r.get("yaw_mean_abs_rate"),
    "yaw_saturation_fraction": r.get("yaw_saturation_fraction"),
    "reason": r.get("reason"),
}, indent=2, sort_keys=True))
PY

  RISK_JSON="$NEXT_RISK_JSON"
done

echo
echo "============================================================"
echo "[TRACER] adaptive theta summary"
echo "============================================================"
column -s, -t < "$SUMMARY_CSV" || cat "$SUMMARY_CSV"

echo
echo "[TRACER] outputs:"
echo "  run_root: $RUN_ROOT"
echo "  summary:  $SUMMARY_CSV"
