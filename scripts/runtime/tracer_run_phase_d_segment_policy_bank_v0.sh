#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORLD="${TRACER_REPEAT_WORLD:-tracer_mixed_stress_course_v5_lowfric_from_solid}"
GOAL_X="${TRACER_REPEAT_GOAL_X:-8.0}"
LATERAL_BOUND="${TRACER_REPEAT_LATERAL_BOUND:-2.0}"
N="${TRACER_REPEAT_N:-1}"
TIMEOUT_S="${TRACER_REPEAT_TIMEOUT_S:-460}"
OPEN_GUI="${TRACER_REPEAT_OPEN_GUI:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="reports/phase_d_segment_policy_${STAMP}_manifest.tsv"
mkdir -p reports logs

cat > "$MANIFEST" <<EOF
label	policy	trial	schedule	log_dir
EOF

POLICY_BANK="${TRACER_SEGMENT_POLICY_BANK:-all_base 0,999,0.200,0.320,0.045,base_all}"

echo "[TRACER] Phase-D segment policy bank v0"
echo "  WORLD=$WORLD"
echo "  GOAL_X=$GOAL_X"
echo "  LATERAL_BOUND=$LATERAL_BOUND"
echo "  N=$N"
echo "  TIMEOUT_S=$TIMEOUT_S"
echo "  OPEN_GUI=$OPEN_GUI"
echo "  MANIFEST=$MANIFEST"
echo
echo "[TRACER] policies:"
echo "$POLICY_BANK"
echo

while IFS= read -r line; do
  [[ -z "${line// }" ]] && continue
  [[ "$line" =~ ^# ]] && continue

  label="$(awk '{print $1}' <<< "$line")"
  schedule="${line#"$label"}"
  schedule="${schedule#"${schedule%%[![:space:]]*}"}"

  for trial in $(seq 1 "$N"); do
    echo
    echo "============================================================"
    echo "[TRACER] RUN label=$label trial=$trial"
    echo "[TRACER] schedule=$schedule"
    echo "============================================================"

    bash scripts/runtime/tracer_stop_phase_c_mixed_stack_v0.sh || true

    export TRACER_PHASE_C_WORLD="$WORLD"
    export TRACER_PHASE_C_GOAL_X="$GOAL_X"
    export TRACER_OPEN_GZCLIENT="$OPEN_GUI"
    export TRACER_PHASE_D_SEGMENT_POLICY_LABEL="$label"
    export TRACER_PHASE_D_SEGMENT_SCHEDULE="$schedule"

    bash scripts/runtime/tracer_start_phase_d_segment_action_stack_v0.sh

    sleep 2
    LOG_DIR="$(ls -td "$ROOT"/logs/phase_d_segment_action_* 2>/dev/null | head -1 || true)"
    if [[ -z "$LOG_DIR" ]]; then
      echo "[TRACER] ERROR: cannot find phase_d_segment_action log dir"
      exit 1
    fi

    echo -e "${label}\tsegment\t${trial}\t${schedule}\t${LOG_DIR}" >> "$MANIFEST"
    echo "[TRACER] monitor: $label log_dir=$LOG_DIR"

    start_ts="$(date +%s)"
    last_x="0"
    last_y="0"

    while true; do
      sleep 10
      elapsed=$(( $(date +%s) - start_ts ))

      parsed="$(
        python3 - "$LOG_DIR/segment_action.log" <<'PY'
import sys, re, pathlib
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print("0 0 False NONE")
    raise SystemExit
lines = p.read_text(errors="ignore").splitlines()
pat = re.compile(r"seq=(\d+).*?x=([-+0-9.]+).*?y=([-+0-9.]+).*?seg=([^ ]+).*?stopped=(True|False)")
for line in reversed(lines):
    m = pat.search(line)
    if m:
        print(m.group(2), m.group(3), m.group(5), m.group(4))
        raise SystemExit
print("0 0 False NONE")
PY
      )"

      read -r last_x last_y stopped seg_name <<< "$parsed"

      echo "[TRACER] t=${elapsed}s label=${label} seg=${seg_name} x=${last_x} y=${last_y} stopped=${stopped}"

      goal_crossed="$(python3 - <<PY
x=float("$last_x")
goal=float("$GOAL_X")
print("1" if x >= goal else "0")
PY
)"
      out_lane="$(python3 - <<PY
y=abs(float("$last_y"))
bound=float("$LATERAL_BOUND")
print("1" if y > bound else "0")
PY
)"
      startup_failed="$(python3 - <<PY
x=abs(float("$last_x"))
elapsed=float("$elapsed")
print("1" if elapsed >= 60 and x < 0.20 else "0")
PY
)"

      if [[ "$startup_failed" == "1" ]]; then
        echo "[TRACER] startup_failed: x<0.20 after ${elapsed}s"
        break
      fi

      if [[ "$out_lane" == "1" ]]; then
        echo "[TRACER] out_of_lane"
        break
      fi

      if [[ "$goal_crossed" == "1" || "$stopped" == "True" ]]; then
        echo "[TRACER] goal crossed; wait 8s for stop logging"
        sleep 8
        break
      fi

      if [[ "$elapsed" -ge "$TIMEOUT_S" ]]; then
        echo "[TRACER] timeout"
        break
      fi
    done

    bash scripts/runtime/tracer_stop_phase_c_mixed_stack_v0.sh || true
  done
done <<< "$POLICY_BANK"

echo
echo "============================================================"
echo "[TRACER] all segment-policy runs finished."
echo "============================================================"
echo "MANIFEST: $MANIFEST"
