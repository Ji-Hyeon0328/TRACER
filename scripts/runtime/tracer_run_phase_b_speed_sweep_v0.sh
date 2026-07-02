#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT="${TRACER_PHASE_B_SWEEP_ROOT:-$ROOT/artifacts/phase_b_speed_sweep_${STAMP}}"

WORLD="${TRACER_PHASE_B_WORLD:-earth}"
SAMPLE_HZ="${TRACER_PHASE_B_SAMPLE_HZ:-20.0}"
STOP_DISTANCE="${TRACER_PHASE_B_GOAL_STOP_DISTANCE:-0.15}"

mkdir -p "$SWEEP_ROOT"

echo "[TRACER] Phase-B speed sweep"
echo "[TRACER] root:       $ROOT"
echo "[TRACER] sweep_root: $SWEEP_ROOT"
echo "[TRACER] world:      $WORLD"

cd "$ROOT"

CONFIGS=(
  "g05_safe_v009_n004_s025 0.5 30.0 0.09 0.04 0.25"
  "g05_mid_v012_n006_s020 0.5 30.0 0.12 0.06 0.20"
  "g05_fast_v014_n007_s020 0.5 30.0 0.14 0.07 0.20"
  "g05_fast_v016_n008_s020 0.5 30.0 0.16 0.08 0.20"
  "g03_mid_v012_n006_s020 0.3 20.0 0.12 0.06 0.20"
  "g08_mid_v012_n006_s025 0.8 40.0 0.12 0.06 0.25"
)

for cfg in "${CONFIGS[@]}"; do
  read -r NAME GOAL DURATION VX_FAR VX_NEAR SLOW_DISTANCE <<< "$cfg"

  RUN_DIR="$SWEEP_ROOT/$NAME"
  mkdir -p "$RUN_DIR"

  echo
  echo "============================================================"
  echo "[TRACER] run: $NAME"
  echo "[TRACER] goal=$GOAL duration=$DURATION vx_far=$VX_FAR vx_near=$VX_NEAR slow=$SLOW_DISTANCE"
  echo "============================================================"

  if TRACER_PHASE_B_WORLD="$WORLD" \
     TRACER_PHASE_B_RECORD_DURATION="$DURATION" \
     TRACER_PHASE_B_SAMPLE_HZ="$SAMPLE_HZ" \
     TRACER_PHASE_B_GOAL_DISTANCE_AHEAD="$GOAL" \
     TRACER_PHASE_B_VX_FAR="$VX_FAR" \
     TRACER_PHASE_B_VX_NEAR="$VX_NEAR" \
     TRACER_PHASE_B_GOAL_SLOW_DISTANCE="$SLOW_DISTANCE" \
     TRACER_PHASE_B_GOAL_STOP_DISTANCE="$STOP_DISTANCE" \
     TRACER_PHASE_B_RUN_DIR="$RUN_DIR" \
     "$ROOT/scripts/runtime/tracer_run_phase_b_goal_episode_v0.sh" \
       2>&1 | tee "$RUN_DIR/episode_runner.log"; then
    echo "[TRACER] run completed: $NAME"
  else
    echo "[TRACER][WARN] run failed: $NAME"
    echo "$NAME" >> "$SWEEP_ROOT/failed_runs.txt"
  fi
done

echo
echo "============================================================"
echo "[TRACER] aggregate sweep"
echo "============================================================"

/usr/bin/python3 "$ROOT/scripts/runtime/tracer_aggregate_phase_b_sweep_v0.py" \
  --sweep-root "$SWEEP_ROOT" \
  --out-csv "$SWEEP_ROOT/sweep_summary.csv" \
  --out-json "$SWEEP_ROOT/sweep_summary.json" \
  --out-prefs "$SWEEP_ROOT/preference_pairs.csv" \
  | tee "$SWEEP_ROOT/aggregate.log"

echo
echo "[TRACER] sweep outputs:"
echo "  $SWEEP_ROOT/sweep_summary.csv"
echo "  $SWEEP_ROOT/sweep_summary.json"
echo "  $SWEEP_ROOT/preference_pairs.csv"
echo "  $SWEEP_ROOT/aggregate.log"
