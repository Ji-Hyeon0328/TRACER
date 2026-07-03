#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

TABLE_JSON="${TRACER_PHASE_B_THETA5_TABLE_JSON:-configs/phase_b_theta5_teacher_table_v0/best_table.json}"
REPEATS="${TRACER_PHASE_B_REPLAY_REPEATS:-2}"
DURATION="${TRACER_PHASE_B_RECORD_DURATION:-30.0}"
SAMPLE_HZ="${TRACER_PHASE_B_SAMPLE_HZ:-20.0}"
KEYS="${TRACER_PHASE_B_REPLAY_KEYS:-earth::low tracer_sponge_firm_flat::low tracer_sponge_firm_slope_5deg::low tracer_sponge_firm_downslope_5deg::low}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${TRACER_PHASE_B_REPLAY_RUN_ROOT:-$ROOT/artifacts/phase_b_theta5_teacher_replay_eval_${STAMP}}"
mkdir -p "$RUN_ROOT"

SUMMARY_CSV="$RUN_ROOT/replay_eval_summary.csv"

echo "[TRACER] theta5 teacher replay eval"
echo "[TRACER] table:    $TABLE_JSON"
echo "[TRACER] repeats:  $REPEATS"
echo "[TRACER] duration: $DURATION"
echo "[TRACER] keys:     $KEYS"
echo "[TRACER] run_root: $RUN_ROOT"

echo "key,world,risk,repeat,case_name,vx_far,vx_near,body_height,swing_clearance,reached,min_rel_dist,final_rel_dist,progress,odom_x_delta,max_yaw,run_dir,summary_json" \
  > "$SUMMARY_CSV"

for KEY in $KEYS; do
  WORLD="${KEY%%::*}"
  RISK="${KEY##*::}"

  ACTION_JSON="$RUN_ROOT/action_${WORLD}_${RISK}.json"

  python3 - "$TABLE_JSON" "$KEY" "$ACTION_JSON" <<'PY'
import json
import sys
from pathlib import Path

table_path, key, out_path = sys.argv[1:]

with open(table_path, "r") as f:
    t = json.load(f)

entry = t["best_table"].get(key)
if entry is None:
    raise SystemExit(f"missing table key: {key}")

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(entry, f, indent=2, sort_keys=True)

print(json.dumps(entry, indent=2, sort_keys=True))
PY

  for REP in $(seq 1 "$REPEATS"); do
    CASE_NAME="$(python3 - "$ACTION_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("case_name", "unknown_case"))
PY
)"
    VX_FAR="$(python3 - "$ACTION_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["action"]["vx_far"])
PY
)"
    VX_NEAR="$(python3 - "$ACTION_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["action"]["vx_near"])
PY
)"
    SLOW="$(python3 - "$ACTION_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["action"]["goal_slow_distance"])
PY
)"
    BODY_H="$(python3 - "$ACTION_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["action"]["body_height"])
PY
)"
    CLR="$(python3 - "$ACTION_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["action"]["swing_clearance"])
PY
)"

    CASE_DIR="$RUN_ROOT/${WORLD}_${RISK}_r${REP}"

    echo
    echo "============================================================"
    echo "[TRACER] replay key=$KEY repeat=$REP case=$CASE_NAME"
    echo "[TRACER] vx=$VX_FAR h=$BODY_H clr=$CLR"
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
      python3 - "$KEY" "$WORLD" "$RISK" "$REP" "$CASE_NAME" "$CASE_DIR" "$SUMMARY_JSON" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys

key, world, risk, rep, case_name, run_dir, summary_json, summary_csv = sys.argv[1:]

with open(summary_json, "r") as f:
    s = json.load(f)

row = {
    "key": key,
    "world": world,
    "risk": risk,
    "repeat": int(rep),
    "case_name": case_name,
    "vx_far": s.get("vx_far"),
    "vx_near": s.get("vx_near"),
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
done

echo
echo "============================================================"
echo "[TRACER] replay eval summary"
echo "============================================================"
column -s, -t < "$SUMMARY_CSV" || cat "$SUMMARY_CSV"

echo
echo "[TRACER] outputs:"
echo "  run_root: $RUN_ROOT"
echo "  summary:  $SUMMARY_CSV"
