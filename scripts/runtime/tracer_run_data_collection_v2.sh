#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"

TERRAINS="${TRACER_DATA_V2_TERRAINS:-flat_normal rough_mid slope_5deg}"
REPEATS="${TRACER_DATA_V2_REPEATS:-2}"
DURATION="${TRACER_DATA_V2_DURATION_SEC:-30.0}"
SAMPLE_HZ="${TRACER_DATA_V2_SAMPLE_HZ:-5.0}"
POLICY_ID_PREFIX="${TRACER_DATA_V2_POLICY_ID_PREFIX:-theta_mlp_udp_v0_data_v2}"
OUT_DIR="${TRACER_DATA_V2_SHADOW_OUT_DIR:-$ROOT/data/shadow_logs_v2_data}"

cd "$ROOT"

echo "[TRACER] data collection v2"
echo "[TRACER] terrains:  $TERRAINS"
echo "[TRACER] repeats:   $REPEATS"
echo "[TRACER] duration:  $DURATION"
echo "[TRACER] sample_hz: $SAMPLE_HZ"
echo "[TRACER] shadow:    $OUT_DIR"

mkdir -p "$OUT_DIR"

if ! ss -lunp | grep -q ':50410'; then
  echo "[TRACER][WARN] supervised stack UDP server is not listening on :50410"
  echo "[TRACER][WARN] start it with:"
  echo "  conda activate env_isaaclab && scripts/runtime/tracer_start_supervised_stack_udp_server_v1.sh"
fi

for terrain in $TERRAINS; do
  for rep in $(seq 1 "$REPEATS"); do
    echo
    echo "========== DATA_V2 terrain=$terrain rep=$rep =========="

    shadow_duration=$(python3 - <<PY
d = float("$DURATION")
print(d + 18.0)
PY
)

    TRACER_SHADOW_OUT_DIR="$OUT_DIR" \
    scripts/runtime/tracer_start_learned_stack_shadow_v2.sh "$terrain" "$shadow_duration"

    TRACER_ROLLOUT_TERRAINS="$terrain" \
    TRACER_ROLLOUT_REPEATS=1 \
    TRACER_ROLLOUT_DURATION_SEC="$DURATION" \
    TRACER_ROLLOUT_SAMPLE_HZ="$SAMPLE_HZ" \
    TRACER_POLICY_ID="${POLICY_ID_PREFIX}_${terrain}_r${rep}" \
    TRACER_META_GAIT_POLICY_KIND=udp \
    TRACER_OBJECTIVE_SELECTOR_VERBOSE=0 \
    scripts/runtime/tracer_run_policy_rollout_dataset_v0.sh
  done
done

echo
echo "[TRACER] summarizing shadow v2 data"
python3 scripts/training/tracer_summarize_learned_stack_shadow_v2.py \
  --glob "data/shadow_logs_v2_data/learned_stack_shadow_v2_*.csv" \
  --out-json data/shadow_logs_v2_data/learned_stack_shadow_v2_data_summary.json

echo
echo "[TRACER] data collection v2 done"
