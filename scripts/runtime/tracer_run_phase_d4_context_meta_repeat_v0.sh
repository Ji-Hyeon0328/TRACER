#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
WORLD="${TRACER_D4_REPEAT_WORLD:-tracer_mixed_stress_course_v5_lowfric_from_solid}"
GOAL_X="${TRACER_D4_REPEAT_GOAL_X:-8.0}"
LATERAL_BOUND="${TRACER_D4_REPEAT_LATERAL_BOUND:-2.0}"
N="${TRACER_D4_REPEAT_N:-3}"
TIMEOUT_S="${TRACER_D4_REPEAT_TIMEOUT_S:-260}"
HOLD_OBS_S="${TRACER_D4_HOLD_OBS_S:-60}"
SLEEP_S="${TRACER_D4_REPEAT_SLEEP_S:-10}"
OPEN_GUI="${TRACER_D4_REPEAT_OPEN_GUI:-0}"

STOP_MARGIN="${TRACER_PHASE_D4_STOP_MARGIN:--0.05}"
YAW_SIGN="${TRACER_PHASE_D4_YAW_SIGN:-0.0}"
YAW_K="${TRACER_PHASE_D4_YAW_K:-0.0}"
YAW_MAX="${TRACER_PHASE_D4_YAW_MAX:-0.0}"

ACTION_TABLE="${TRACER_PHASE_D4_CONTEXT_ACTION_TABLE:-flat:0.210,0.320,0.045;start_flat:0.210,0.320,0.045;rough:0.210,0.320,0.045;upslope:0.210,0.320,0.045;downslope:0.2025,0.320,0.045;goal_flat:0.2025,0.320,0.045;unknown:0.2025,0.320,0.045}"

TS="$(date +%Y%m%d_%H%M%S)"
MANIFEST="$ROOT/reports/phase_d4_context_meta_repeat_${TS}_manifest.tsv"

mkdir -p "$ROOT/reports"

echo -e "label\ttrial\tworld\tgoal_x\tlateral_bound\tlog_dir\ttimeout_s\thold_obs_s" > "$MANIFEST"

echo "[TRACER] Phase-D4 context-meta repeat v0"
echo "  WORLD=$WORLD"
echo "  GOAL_X=$GOAL_X"
echo "  LATERAL_BOUND=$LATERAL_BOUND"
echo "  N=$N"
echo "  TIMEOUT_S=$TIMEOUT_S"
echo "  HOLD_OBS_S=$HOLD_OBS_S"
echo "  MANIFEST=$MANIFEST"
echo

cd "$ROOT"

for trial in $(seq 1 "$N"); do
  label="d4_context_meta_v0"

  echo
  echo "============================================================"
  echo "[TRACER] RUN label=$label trial=$trial"
  echo "============================================================"

  bash scripts/runtime/tracer_stop_phase_c_mixed_stack_v0.sh || true
  pkill -f "tracer_phase_d4_context_meta_selector_node_v0.py" 2>/dev/null || true

  TRACER_PHASE_C_WORLD="$WORLD" \
  TRACER_PHASE_D4_GOAL_X="$GOAL_X" \
  TRACER_PHASE_D4_STOP_MARGIN="$STOP_MARGIN" \
  TRACER_PHASE_D4_LATERAL_BOUND="$LATERAL_BOUND" \
  TRACER_PHASE_D4_YAW_SIGN="$YAW_SIGN" \
  TRACER_PHASE_D4_YAW_K="$YAW_K" \
  TRACER_PHASE_D4_YAW_MAX="$YAW_MAX" \
  TRACER_PHASE_D4_CONTEXT_ACTION_TABLE="$ACTION_TABLE" \
  TRACER_OPEN_GZCLIENT="$OPEN_GUI" \
  bash scripts/runtime/tracer_start_phase_d4_context_meta_stack_v0.sh

  LOG_DIR="$(ls -td "$ROOT"/logs/phase_d4_context_meta_* | head -1)"
  echo -e "${label}\t${trial}\t${WORLD}\t${GOAL_X}\t${LATERAL_BOUND}\t${LOG_DIR}\t${TIMEOUT_S}\t${HOLD_OBS_S}" >> "$MANIFEST"

  echo "[TRACER] monitor: $LOG_DIR"

  reached_at=""
  start_t="$(date +%s)"

  while true; do
    now="$(date +%s)"
    elapsed=$((now - start_t))

    STATE="$(python3 - "$LOG_DIR/context_meta_selector.log" "$GOAL_X" "$LATERAL_BOUND" <<'PY'
import sys, re, math
path, goal_x, lateral = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
pat = re.compile(r"seq=(?P<seq>\d+).*?context=(?P<context>\S+).*?x=(?P<x>-?\d+\.\d+).*?y=(?P<y>-?\d+\.\d+).*?out_of_lane=(?P<out>\S+).*?startup_failed=(?P<startup>\S+).*?stopped=(?P<stopped>\S+)")
last = None
reached = False
out_lane = False
startup = False
try:
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            d = m.groupdict()
            x = float(d["x"])
            y = float(d["y"])
            reached = reached or (x >= goal_x)
            out_lane = out_lane or (abs(y) > lateral) or (d["out"] == "True")
            startup = startup or (d["startup"] == "True")
            last = (int(d["seq"]), d["context"], x, y, d["stopped"])
except FileNotFoundError:
    pass

if last is None:
    print("NA NA NA False False False False")
else:
    seq, context, x, y, stopped = last
    print(f"{seq} {context} {x:.3f} {y:.3f} {reached} {out_lane} {startup} {stopped}")
PY
)"

    echo "[TRACER] t=${elapsed}s state=${STATE}"

    read -r seq context x y reached out_lane startup stopped <<< "$STATE"

    if [ "$startup" = "True" ] || [ "$out_lane" = "True" ]; then
      echo "[TRACER] fail condition reached: startup=$startup out_lane=$out_lane"
      break
    fi

    if [ "$reached" = "True" ] && [ -z "$reached_at" ]; then
      reached_at="$elapsed"
      echo "[TRACER] reached goal region at t=${reached_at}s; observing hold for ${HOLD_OBS_S}s"
    fi

    if [ -n "$reached_at" ] && [ $((elapsed - reached_at)) -ge "$HOLD_OBS_S" ]; then
      echo "[TRACER] hold observation done"
      break
    fi

    if [ "$elapsed" -ge "$TIMEOUT_S" ]; then
      echo "[TRACER] timeout"
      break
    fi

    sleep "$SLEEP_S"
  done

  bash scripts/runtime/tracer_stop_phase_c_mixed_stack_v0.sh || true
  pkill -f "tracer_phase_d4_context_meta_selector_node_v0.py" 2>/dev/null || true
done

echo
echo "============================================================"
echo "[TRACER] all D4 repeat runs finished."
echo "============================================================"
echo "MANIFEST: $MANIFEST"
