#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

MODEL="${TRACER_PHASE_B_SELECTOR_MODEL:-}"
RISK_STATE_JSON="${TRACER_PHASE_B_RISK_STATE_JSON:-}"
WORLD="${TRACER_PHASE_B_WORLD:-earth}"

if [[ -z "$MODEL" ]]; then
  MODEL="$(ls -td artifacts/phase_b_speed_selector_v0_* 2>/dev/null | head -1)/phase_b_speed_selector_v0.json"
fi

if [[ ! -s "$MODEL" ]]; then
  echo "[TRACER][ERROR] selector model not found: $MODEL"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${TRACER_PHASE_B_RUN_DIR:-$ROOT/artifacts/phase_b_selected_goal_episode_${STAMP}}"
mkdir -p "$RUN_DIR"

SELECTION_JSON="$RUN_DIR/selected_speed_profile.json"

echo "[TRACER] selected Phase-B goal episode"
echo "[TRACER] root:       $ROOT"
echo "[TRACER] model:      $MODEL"
echo "[TRACER] risk_json:  ${RISK_STATE_JSON:-<default-low-risk>}"
echo "[TRACER] run_dir:    $RUN_DIR"

if [[ -n "$RISK_STATE_JSON" ]]; then
  python3 "$ROOT/scripts/runtime/tracer_select_phase_b_speed_profile_v0.py" \
    --model "$MODEL" \
    --risk-state-json "$RISK_STATE_JSON" \
    --out-json "$SELECTION_JSON" \
    > "$RUN_DIR/selector_stdout.json"
else
  python3 "$ROOT/scripts/runtime/tracer_select_phase_b_speed_profile_v0.py" \
    --model "$MODEL" \
    --out-json "$SELECTION_JSON" \
    > "$RUN_DIR/selector_stdout.json"
fi

echo
echo "========== selected profile =========="
python3 - "$SELECTION_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    s = json.load(f)

sel = s["selected"]
p = sel["profile"]

print(json.dumps({
    "name": sel["name"],
    "score": sel["score"],
    "allowed": sel["allowed"],
    "guard_reason": sel["guard_reason"],
    "profile": p,
    "risk_state": s.get("risk_state", {}),
}, indent=2, sort_keys=True))
PY

read -r SELECTED_NAME VX_FAR VX_NEAR SLOW_DISTANCE STOP_DISTANCE GOAL_DISTANCE <<< "$(
python3 - "$SELECTION_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    s = json.load(f)

p = s["selected"]["profile"]
print(
    s["selected"]["name"],
    p["vx_far"],
    p["vx_near"],
    p["goal_slow_distance"],
    p["goal_stop_distance"],
    p["goal_distance_ahead"],
)
PY
)"

echo
echo "[TRACER] selected_name:  $SELECTED_NAME"
echo "[TRACER] vx_far:         $VX_FAR"
echo "[TRACER] vx_near:        $VX_NEAR"
echo "[TRACER] slow_distance:  $SLOW_DISTANCE"
echo "[TRACER] stop_distance:  $STOP_DISTANCE"
echo "[TRACER] goal_distance:  $GOAL_DISTANCE"

# Preserve user overrides for duration/sample rate, but use selected profile for speed params.
TRACER_PHASE_B_WORLD="$WORLD" \
TRACER_PHASE_B_RUN_DIR="$RUN_DIR/episode" \
TRACER_PHASE_B_GOAL_DISTANCE_AHEAD="$GOAL_DISTANCE" \
TRACER_PHASE_B_VX_FAR="$VX_FAR" \
TRACER_PHASE_B_VX_NEAR="$VX_NEAR" \
TRACER_PHASE_B_GOAL_SLOW_DISTANCE="$SLOW_DISTANCE" \
TRACER_PHASE_B_GOAL_STOP_DISTANCE="$STOP_DISTANCE" \
"$ROOT/scripts/runtime/tracer_run_phase_b_goal_episode_v0.sh" \
  2>&1 | tee "$RUN_DIR/selected_episode_runner.log"

EP_SUMMARY="$RUN_DIR/episode/phase_b_summary.json"

if [[ -s "$EP_SUMMARY" ]]; then
  python3 - "$SELECTION_JSON" "$EP_SUMMARY" <<'PY'
import json
import sys

selection_path, summary_path = sys.argv[1], sys.argv[2]

with open(selection_path, "r") as f:
    selection = json.load(f)

with open(summary_path, "r") as f:
    summary = json.load(f)

summary["selector_model"] = selection.get("model")
summary["selected_profile_name"] = selection["selected"]["name"]
summary["selected_profile_score"] = selection["selected"]["score"]
summary["selected_profile_guard_reason"] = selection["selected"]["guard_reason"]
summary["selected_profile_allowed"] = selection["selected"]["allowed"]
summary["risk_state"] = selection.get("risk_state", {})

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

print(json.dumps({
    "episode_summary": summary_path,
    "selected_profile_name": summary["selected_profile_name"],
    "reached_stop_distance": summary.get("reached_stop_distance"),
    "min_rel_dist": summary.get("min_rel_dist"),
    "progress_initial_minus_min": summary.get("progress_initial_minus_min"),
    "odom_x_delta": summary.get("odom_x_delta"),
}, indent=2, sort_keys=True))
PY
else
  echo "[TRACER][WARN] episode summary missing: $EP_SUMMARY"
fi

echo
echo "[TRACER] outputs:"
echo "  selection: $SELECTION_JSON"
echo "  episode:   $RUN_DIR/episode"
echo "  summary:   $EP_SUMMARY"
