#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

GAZEBO_DOCKER="${TRACER_GAZEBO_DOCKER:-a1_unitree_gazebo_docker}"

TERRAINS="${TRACER_PHASE_B_HC_TERRAINS:-earth tracer_sponge_firm_flat tracer_sponge_firm_slope_5deg tracer_sponge_firm_downslope_5deg}"
DURATION="${TRACER_PHASE_B_RECORD_DURATION:-30.0}"
SAMPLE_HZ="${TRACER_PHASE_B_SAMPLE_HZ:-20.0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${TRACER_PHASE_B_HC_TERRAIN_SET_RUN_ROOT:-$ROOT/artifacts/phase_b_theta_hc_sweep_terrain_set_${STAMP}}"
mkdir -p "$RUN_ROOT"

MANIFEST="$RUN_ROOT/terrain_set_manifest.csv"

echo "[TRACER] Phase-B theta HC terrain-set sweep"
echo "[TRACER] root:      $ROOT"
echo "[TRACER] terrains:  $TERRAINS"
echo "[TRACER] duration:  $DURATION"
echo "[TRACER] sample_hz: $SAMPLE_HZ"
echo "[TRACER] run_root:  $RUN_ROOT"

echo "world,status,run_root,summary_csv" > "$MANIFEST"

for WORLD in $TERRAINS; do
  echo
  echo "============================================================"
  echo "[TRACER] terrain HC sweep: $WORLD"
  echo "============================================================"

  if [[ "$WORLD" != "earth" ]]; then
    if ! docker exec "$GAZEBO_DOCKER" bash -lc "test -f /root/unitree_ws/src/unitree_ros/unitree_gazebo/worlds/${WORLD}.world" >/dev/null 2>&1; then
      echo "[TRACER][WARN] world file not found in Gazebo container, skipping: $WORLD"
      echo "$WORLD,skipped_missing_world,," >> "$MANIFEST"
      continue
    fi
  fi

  TERRAIN_RUN_ROOT="$RUN_ROOT/$WORLD"

  set +e
  TRACER_PHASE_B_WORLD="$WORLD" \
  TRACER_PHASE_B_RECORD_DURATION="$DURATION" \
  TRACER_PHASE_B_SAMPLE_HZ="$SAMPLE_HZ" \
  TRACER_PHASE_B_HC_SWEEP_RUN_ROOT="$TERRAIN_RUN_ROOT" \
  "$ROOT/scripts/runtime/tracer_run_phase_b_theta_height_clearance_sweep_v0.sh" \
    2>&1 | tee "$RUN_ROOT/${WORLD}.log"
  RC="${PIPESTATUS[0]}"
  set -e

  if [[ "$RC" -ne 0 ]]; then
    echo "[TRACER][WARN] sweep failed for world=$WORLD rc=$RC"
    echo "$WORLD,failed,$TERRAIN_RUN_ROOT," >> "$MANIFEST"
    continue
  fi

  SUMMARY_CSV="$TERRAIN_RUN_ROOT/hc_sweep_summary.csv"
  if [[ -s "$SUMMARY_CSV" ]]; then
    echo "$WORLD,ok,$TERRAIN_RUN_ROOT,$SUMMARY_CSV" >> "$MANIFEST"
  else
    echo "$WORLD,missing_summary,$TERRAIN_RUN_ROOT," >> "$MANIFEST"
  fi
done

echo
echo "============================================================"
echo "[TRACER] terrain-set manifest"
echo "============================================================"
column -s, -t < "$MANIFEST" || cat "$MANIFEST"

echo
echo "[TRACER] outputs:"
echo "  run_root: $RUN_ROOT"
echo "  manifest: $MANIFEST"
