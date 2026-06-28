#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
TERRAINS="${TRACER_ROLLOUT_TERRAINS:-flat_normal rough_mid slope_5deg}"
REPEATS="${TRACER_ROLLOUT_REPEATS:-1}"
DURATION="${TRACER_ROLLOUT_DURATION_SEC:-12.0}"
SAMPLE_HZ="${TRACER_ROLLOUT_SAMPLE_HZ:-5.0}"
POLICY_ID="${TRACER_POLICY_ID:-world_reload_rollout_v0}"

cd "$ROOT"

terrain_to_world() {
  local terrain="$1"
  case "$terrain" in
    flat_normal)
      echo "${TRACER_WORLD_FLAT_NORMAL:-stairs_single}"
      ;;
    rough_mid)
      echo "${TRACER_WORLD_ROUGH_MID:-tracer_rough_mid}"
      ;;
    rough_low)
      echo "${TRACER_WORLD_ROUGH_LOW:-tracer_rough_low}"
      ;;
    slope_5deg)
      echo "${TRACER_WORLD_SLOPE_5DEG:-tracer_slope_5deg}"
      ;;
    downslope_5deg|slippery_downslope_5deg_forward)
      echo "${TRACER_WORLD_DOWNSLOPE_5DEG:-tracer_downslope_5deg}"
      ;;
    sponge_firm_slope_5deg)
      echo "${TRACER_WORLD_SPONGE_FIRM_SLOPE_5DEG:-tracer_sponge_firm_slope_5deg}"
      ;;
    sponge_firm_downslope_5deg|sponge_firm_downslope_5deg_forward)
      echo "${TRACER_WORLD_SPONGE_FIRM_DOWNSLOPE_5DEG:-tracer_sponge_firm_downslope_5deg}"
      ;;
    *)
      echo "${TRACER_WORLD_DEFAULT:-stairs_single}"
      ;;
  esac
}

echo "[TRACER] world-aware rollout dataset runner"
echo "[TRACER] terrains:  $TERRAINS"
echo "[TRACER] repeats:   $REPEATS"
echo "[TRACER] duration:  $DURATION"
echo "[TRACER] sample_hz: $SAMPLE_HZ"
echo "[TRACER] policy_id: $POLICY_ID"

for repeat in $(seq 1 "$REPEATS"); do
  for terrain in $TERRAINS; do
    world="$(terrain_to_world "$terrain")"

    echo
    echo "========== WORLD-AWARE TERRAIN=$terrain WORLD=$world REPEAT=$repeat =========="

    scripts/runtime/tracer_launch_gazebo_world_split_safe.sh "$world"

    echo "[TRACER] reset to qwer state"
    scripts/runtime/tracer_reset_to_qwer_state.sh

    echo "[TRACER] start qwerty bridge/state"
    scripts/runtime/tracer_start_qwerty_state.sh

    echo "[TRACER] restart A1-QP-MPC controller split-safe"
    scripts/runtime/tracer_restart_a1_qpmc_controller_split_safe.sh

    echo "[TRACER] run one-terrain rollout"
    TRACER_ROLLOUT_TERRAINS="$terrain" \
    TRACER_ROLLOUT_REPEATS=1 \
    TRACER_ROLLOUT_DURATION_SEC="$DURATION" \
    TRACER_ROLLOUT_SAMPLE_HZ="$SAMPLE_HZ" \
    TRACER_POLICY_ID="${POLICY_ID}_world_${world}" \
    scripts/runtime/tracer_run_policy_rollout_dataset_v0.sh
  done
done

echo
echo "[TRACER] world-aware rollout finished"
