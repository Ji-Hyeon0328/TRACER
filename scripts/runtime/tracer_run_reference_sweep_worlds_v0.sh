#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/rollout/tracer_reference_sweep_phase_a_mini_terrain_v0.yaml}"
OUT_DIR="${2:-data/rollouts/reference_sweep_v0}"
DURATION_SEC="${TRACER_SWEEP_DURATION_SEC:-5.0}"
REPEATS="${TRACER_SWEEP_REPEATS:-3}"
SAMPLE_HZ="${TRACER_SWEEP_SAMPLE_HZ:-10.0}"
PUBLISH_HZ="${TRACER_SWEEP_PUBLISH_HZ:-20.0}"

# terrain_name:world_name pairs.
# terrain_name must exist in configs/terrain_dataset/terrain_set_v0.yaml.
TERRAINS=(
  "flat_normal:earth"
  "sponge_firm_flat:tracer_sponge_firm_flat"
  "slippery_mild_flat:tracer_slippery_mild_flat"
)

echo "[TRACER] reference sweep worlds v0"
echo "[TRACER] config:      $CONFIG"
echo "[TRACER] out_dir:     $OUT_DIR"
echo "[TRACER] duration:    $DURATION_SEC"
echo "[TRACER] repeats:     $REPEATS"
echo "[TRACER] sample_hz:   $SAMPLE_HZ"
echo "[TRACER] publish_hz:  $PUBLISH_HZ"

for pair in "${TERRAINS[@]}"; do
  TERRAIN="${pair%%:*}"
  WORLD="${pair##*:}"

  echo
  echo "================================================================================"
  echo "[TRACER] terrain=$TERRAIN world=$WORLD"
  echo "================================================================================"

  scripts/runtime/tracer_launch_gazebo_world_clean.sh "$WORLD"

  /usr/bin/python3 scripts/runtime/tracer_run_reference_sweep_rollouts_v0.py \
    --config "$CONFIG" \
    --terrains "$TERRAIN" \
    --duration-sec "$DURATION_SEC" \
    --sample-hz "$SAMPLE_HZ" \
    --publish-hz "$PUBLISH_HZ" \
    --repeats "$REPEATS" \
    --out-dir "$OUT_DIR" \
    --strict-world-check

  LATEST="$(ls -td "$OUT_DIR"/ref_sweep_v0_* | head -1)"
  echo "[TRACER] latest run: $LATEST"

  python3 scripts/training/tracer_score_reference_sweep_phase_a_v0.py \
    --run-dir "$LATEST" \
    --top-k 3
done

echo
echo "[TRACER] world sweep done."
