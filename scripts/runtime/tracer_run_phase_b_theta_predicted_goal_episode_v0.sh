#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

THETA_MODEL="${TRACER_PHASE_B_THETA_MODEL:-}"
RISK_STATE_JSON="${TRACER_PHASE_B_RISK_STATE_JSON:-}"
WORLD="${TRACER_PHASE_B_WORLD:-earth}"
GOAL_DISTANCE="${TRACER_PHASE_B_GOAL_DISTANCE_AHEAD:-0.5}"
STOP_DISTANCE="${TRACER_PHASE_B_GOAL_STOP_DISTANCE:-0.15}"

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

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${TRACER_PHASE_B_RUN_DIR:-$ROOT/artifacts/phase_b_theta_predicted_goal_episode_${STAMP}}"
mkdir -p "$RUN_DIR"

if [[ -z "$RISK_STATE_JSON" ]]; then
  RISK_STATE_JSON="$RUN_DIR/default_low_risk_state.json"
  cat > "$RISK_STATE_JSON" <<'JSON'
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
fi

PRED_JSON="$RUN_DIR/theta_prediction.json"

echo "[TRACER] theta-predicted Phase-B goal episode"
echo "[TRACER] root:       $ROOT"
echo "[TRACER] model:      $THETA_MODEL"
echo "[TRACER] risk_json:  $RISK_STATE_JSON"
echo "[TRACER] world:      $WORLD"
echo "[TRACER] run_dir:    $RUN_DIR"

python3 "$ROOT/scripts/runtime/tracer_predict_phase_b_theta_profile_v0.py" \
  --model "$THETA_MODEL" \
  --risk-state-json "$RISK_STATE_JSON" \
  --goal-distance-ahead "$GOAL_DISTANCE" \
  --goal-stop-distance "$STOP_DISTANCE" \
  --out-json "$PRED_JSON" \
  > "$RUN_DIR/theta_prediction_stdout.json"

echo
echo "========== theta prediction =========="
python3 - "$PRED_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    d = json.load(f)

print(json.dumps({
    "guard_reasons": d.get("guard_reasons", []),
    "raw_prediction": d.get("raw_prediction", {}),
    "profile": d.get("profile", {}),
}, indent=2, sort_keys=True))
PY

read -r VX_FAR VX_NEAR SLOW_DISTANCE STOP_DISTANCE_USED GOAL_DISTANCE_USED <<< "$(
python3 - "$PRED_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    d = json.load(f)

p = d["profile"]
print(
    p["vx_far"],
    p["vx_near"],
    p["goal_slow_distance"],
    p["goal_stop_distance"],
    p["goal_distance_ahead"],
)
PY
)"

echo
echo "[TRACER] predicted vx_far:        $VX_FAR"
echo "[TRACER] predicted vx_near:       $VX_NEAR"
echo "[TRACER] predicted slow_distance: $SLOW_DISTANCE"
echo "[TRACER] stop_distance:           $STOP_DISTANCE_USED"
echo "[TRACER] goal_distance:           $GOAL_DISTANCE_USED"

TRACER_PHASE_B_WORLD="$WORLD" \
TRACER_PHASE_B_RUN_DIR="$RUN_DIR/episode" \
TRACER_PHASE_B_GOAL_DISTANCE_AHEAD="$GOAL_DISTANCE_USED" \
TRACER_PHASE_B_VX_FAR="$VX_FAR" \
TRACER_PHASE_B_VX_NEAR="$VX_NEAR" \
TRACER_PHASE_B_GOAL_SLOW_DISTANCE="$SLOW_DISTANCE" \
TRACER_PHASE_B_GOAL_STOP_DISTANCE="$STOP_DISTANCE_USED" \
"$ROOT/scripts/runtime/tracer_run_phase_b_goal_episode_v0.sh" \
  2>&1 | tee "$RUN_DIR/theta_episode_runner.log"

EP_SUMMARY="$RUN_DIR/episode/phase_b_summary.json"

if [[ -s "$EP_SUMMARY" ]]; then
  python3 - "$PRED_JSON" "$EP_SUMMARY" <<'PY'
import json
import sys

pred_path, summary_path = sys.argv[1], sys.argv[2]

with open(pred_path, "r") as f:
    pred = json.load(f)

with open(summary_path, "r") as f:
    summary = json.load(f)

p = pred["profile"]

summary["theta_model"] = pred.get("model")
summary["theta_profile_name"] = p.get("name")
summary["theta_guard_reasons"] = pred.get("guard_reasons", [])
summary["theta_raw_prediction"] = pred.get("raw_prediction", {})
summary["theta_projected_prediction"] = pred.get("projected_prediction", {})
summary["theta_risk_state"] = pred.get("risk_state", {})

summary["selected_profile_name"] = p.get("name")
summary["selected_profile_guard_reason"] = "; ".join(pred.get("guard_reasons", []))
summary["selected_profile_score"] = None

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

print(json.dumps({
    "episode_summary": summary_path,
    "theta_profile_name": summary["theta_profile_name"],
    "theta_guard_reasons": summary["theta_guard_reasons"],
    "reached_stop_distance": summary.get("reached_stop_distance"),
    "min_rel_dist": summary.get("min_rel_dist"),
    "final_rel_dist": summary.get("final_rel_dist"),
    "progress_initial_minus_min": summary.get("progress_initial_minus_min"),
    "odom_x_delta": summary.get("odom_x_delta"),
    "max_mpc_vx": summary.get("max_mpc_vx"),
    "max_abs_mpc_yaw_rate": summary.get("max_abs_mpc_yaw_rate"),
}, indent=2, sort_keys=True))
PY
else
  echo "[TRACER][WARN] episode summary missing: $EP_SUMMARY"
fi

echo
echo "[TRACER] outputs:"
echo "  prediction: $PRED_JSON"
echo "  episode:    $RUN_DIR/episode"
echo "  summary:    $EP_SUMMARY"
