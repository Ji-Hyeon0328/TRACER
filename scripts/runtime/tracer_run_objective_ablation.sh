#!/usr/bin/env bash
set -eo pipefail
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
REPEATS="${TRACER_ABLATION_REPEATS:-3}"
DISTANCES="${TRACER_WAYPOINT_DISTANCES:-1.0,2.0,3.0}"
TIMEOUT="${TRACER_MISSION_TIMEOUT_SEC:-90}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${TRACER_ABLATION_OUT_DIR:-$ROOT/data/ablations/objective_${RUN_ID}}"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "[TRACER] Objective ablation runner"
echo "============================================================"
echo "[TRACER] root:      $ROOT"
echo "[TRACER] out dir:   $OUT_DIR"
echo "[TRACER] repeats:   $REPEATS"
echo "[TRACER] distances: $DISTANCES"
echo "[TRACER] timeout:   $TIMEOUT"
echo

cd "$ROOT"

MODES=(
  "balanced:0.3333333333333333,0.3333333333333333,0.3333333333333333"
  "motion:0.60,0.25,0.15"
  "stability:0.20,0.65,0.15"
  "energy:0.20,0.25,0.55"
)

for item in "${MODES[@]}"; do
  LABEL="${item%%:*}"
  WEIGHTS="${item#*:}"

  for REP in $(seq 1 "$REPEATS"); do
    echo
    echo "============================================================"
    echo "[TRACER] Ablation run: label=$LABEL rep=$REP/$REPEATS weights=$WEIGHTS"
    echo "============================================================"

    RUN_LOG="$OUT_DIR/${LABEL}_rep$(printf "%02d" "$REP")_runner.log"

    set +e
    TRACER_OBJECTIVE_LABEL="$LABEL" \
    TRACER_OBJECTIVE_WEIGHTS="$WEIGHTS" \
    TRACER_WAYPOINT_DISTANCES="$DISTANCES" \
    TRACER_MISSION_TIMEOUT_SEC="$TIMEOUT" \
    TRACER_AUTO_STOP_QWERTY=1 \
      scripts/runtime/tracer_run_waypoint_mission_from_reset.sh 2>&1 | tee "$RUN_LOG"
    STATUS="${PIPESTATUS[0]}"
    set -e

    LATEST_SUMMARY="$(ls -t "$ROOT"/data/mission_logs/tracer_mission_summary_*.json 2>/dev/null | head -1 || true)"

    if [ -z "$LATEST_SUMMARY" ]; then
      echo "[TRACER] WARNING: no summary found after run label=$LABEL rep=$REP"
      continue
    fi

    python3 - <<PY
import json, os, shutil

summary_path = "$LATEST_SUMMARY"
out_dir = "$OUT_DIR"
label = "$LABEL"
weights = "$WEIGHTS"
rep = int("$REP")
status = int("$STATUS")
run_log = "$RUN_LOG"

with open(summary_path) as f:
    s = json.load(f)

src_npz = s.get("output_npz", "")

prefix = f"{label}_rep{rep:02d}"
dst_summary = os.path.join(out_dir, prefix + "_summary.json")
dst_npz = os.path.join(out_dir, prefix + "_log.npz")

s["ablation_label"] = label
s["ablation_repeat"] = rep
s["ablation_requested_weights"] = [float(x) for x in weights.split(",")]
s["runner_exit_status"] = status
s["runner_log"] = run_log

if src_npz and os.path.exists(src_npz):
    shutil.copy2(src_npz, dst_npz)
    s["output_npz"] = dst_npz
else:
    s["output_npz"] = src_npz

with open(dst_summary, "w") as f:
    json.dump(s, f, indent=2)

print(f"[TRACER] copied summary: {dst_summary}")
print(f"[TRACER] copied npz:     {dst_npz if os.path.exists(dst_npz) else 'missing'}")
PY

    echo "[TRACER] short pause before next run..."
    sleep 2
  done
done

echo
echo "============================================================"
echo "[TRACER] Summarize objective ablation"
echo "============================================================"
python3 scripts/runtime/tracer_summarize_objective_ablation.py "$OUT_DIR"

echo
echo "============================================================"
echo "[TRACER] Objective ablation finished"
echo "============================================================"
echo "[TRACER] out dir: $OUT_DIR"
