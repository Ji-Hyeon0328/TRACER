#!/usr/bin/env bash
set -eo pipefail

ROOT="${TRACER_ROOT:-$HOME/Tracer/TRACER}"
cd "$ROOT"

POLICY_JSON="${TRACER_MICRO_BRAKE_POLICY_JSON:-$ROOT/configs/highlevel_policy/tracer_fusion_policy_slippery_micro_brake_v3.json}"
WORLD="${TRACER_MICRO_BRAKE_WORLD:-tracer_slippery_mid_slope_5deg}"
TERRAIN="${TRACER_MICRO_BRAKE_TERRAIN:-slippery_micro_brake_v3_downhill_behind_body}"
DURATION="${TRACER_MICRO_BRAKE_DURATION:-12}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PREFIX="${TRACER_MICRO_BRAKE_PREFIX:-slippery_micro_brake_staged_v4_${STAMP}}"

RAM_MONITOR_CSV="${TRACER_RAM_MONITOR_CSV:-$ROOT/data/sanity_results/tracer_ram_monitor_results_v2.csv}"
RAM_GATE_CSV="${TRACER_RAM_GATE_CSV:-$ROOT/data/sanity_results/tracer_ram_gate_monitor_results.csv}"
JOINED_OUT_CSV="${TRACER_JOINED_OUT_CSV:-$ROOT/data/sanity_results/tracer_slippery_micro_brake_staged_v4_joined_${PREFIX}.csv}"

echo "============================================================"
echo "[TRACER] slippery micro-brake staged v4"
echo "============================================================"
echo "[TRACER] policy:     $POLICY_JSON"
echo "[TRACER] world:      $WORLD"
echo "[TRACER] terrain:    $TERRAIN"
echo "[TRACER] duration:   $DURATION"
echo "[TRACER] prefix:     $PREFIX"
echo "[TRACER] joined csv: $JOINED_OUT_CSV"
echo "[TRACER] height ramp: start=0.325 duration=${TRACER_RAMP_BODY_HEIGHT_DURATION:-2.0}"
echo "[TRACER] vx ramp:     start=0.0 delay=${TRACER_RAMP_VX_DELAY:-2.0} duration=${TRACER_RAMP_VX_DURATION:-2.0}"
echo "============================================================"

run_one() {
  local i="$1"
  local tag="${PREFIX}_r${i}"

  echo
  echo "============================================================"
  echo "[TRACER] staged micro-brake case: $tag"
  echo "============================================================"

  TRACER_FUSION_POLICY_JSON="$POLICY_JSON" \
  TRACER_ENABLE_RAM_MONITOR=1 \
  TRACER_ENABLE_RAM_GATE_MONITOR=1 \
  TRACER_RAM_REQUIRE_ODOM=1 \
  TRACER_GATE_SKIP_INITIAL_MSGS="${TRACER_GATE_SKIP_INITIAL_MSGS:-3}" \
  TRACER_RAM_RESULT_CSV="$RAM_MONITOR_CSV" \
  TRACER_RAM_GATE_RESULT_CSV="$RAM_GATE_CSV" \
  TRACER_RAMP_BODY_HEIGHT_ENABLE=1 \
  TRACER_RAMP_BODY_HEIGHT_START="${TRACER_RAMP_BODY_HEIGHT_START:-0.325}" \
  TRACER_RAMP_BODY_HEIGHT_DURATION="${TRACER_RAMP_BODY_HEIGHT_DURATION:-2.0}" \
  TRACER_RAMP_VX_ENABLE=1 \
  TRACER_RAMP_VX_START="${TRACER_RAMP_VX_START:-0.0}" \
  TRACER_RAMP_VX_DELAY="${TRACER_RAMP_VX_DELAY:-2.0}" \
  TRACER_RAMP_VX_DURATION="${TRACER_RAMP_VX_DURATION:-2.0}" \
  "$ROOT/scripts/runtime/tracer_run_fusion_policy_sanity_once.sh" \
    "$TERRAIN" \
    "$WORLD" \
    "$DURATION" \
    "$tag"
}

for i in 1 2 3; do
  run_one "$i"
done

echo
echo "============================================================"
echo "[TRACER] joining staged micro-brake v4"
echo "============================================================"

TRACER_RAM_MONITOR_CSV="$RAM_MONITOR_CSV" \
TRACER_RAM_GATE_CSV="$RAM_GATE_CSV" \
TRACER_JOINED_OUT_CSV="$JOINED_OUT_CSV" \
TRACER_JOIN_FILTER_PREFIX="$PREFIX" \
"$ROOT/scripts/runtime/tracer_join_fusion_ram_gate_results.py"

echo
echo "============================================================"
echo "[TRACER] compact staged v4 table"
echo "============================================================"

PREFIX="$PREFIX" JOINED_OUT_CSV="$JOINED_OUT_CSV" python3 - <<'PY'
import os
from pathlib import Path
import pandas as pd

prefix = os.environ["PREFIX"]
p = Path(os.environ["JOINED_OUT_CSV"])
df = pd.read_csv(p)
df = df[df["sanity_tag"].astype(str).str.startswith(prefix)].copy()

if "delta_x" in df.columns and "delta_y" in df.columns:
    df["net_xy"] = (df["delta_x"] ** 2 + df["delta_y"] ** 2).pow(0.5)
    df["abs_dy"] = df["delta_y"].abs()

cols = [
    "sanity_tag",
    "terrain_key_fusion",
    "style",
    "valid_locomotion_candidate",
    "fallen_relative_z_based",
    "delta_x",
    "delta_y",
    "net_xy",
    "min_rel_z",
    "ram_level_last",
    "unstable_frac",
    "ctrl_ema_tail5_mean_gate",
    "fallen_tail5_mean_gate",
    "sigma_tail5_mean_gate",
]
cols = [c for c in cols if c in df.columns]

if df.empty:
    print(f"[TRACER] no joined rows found for prefix={prefix}")
else:
    print(df[cols].to_string(index=False))
    print()
    print("========== aggregate ==========")
    agg = df.groupby("terrain_key_fusion").agg(
        runs=("sanity_tag", "count"),
        valid_rate=("valid_locomotion_candidate", "mean"),
        fallen_rate=("fallen_relative_z_based", "mean"),
        dx_mean=("delta_x", "mean"),
        abs_dy_mean=("abs_dy", "mean"),
        net_xy_mean=("net_xy", "mean"),
        min_rel_z_mean=("min_rel_z", "mean"),
        min_rel_z_min=("min_rel_z", "min"),
        unstable_frac_mean=("unstable_frac", "mean"),
        ctrl_ema_tail5_mean=("ctrl_ema_tail5_mean_gate", "mean"),
        fallen_tail5_mean=("fallen_tail5_mean_gate", "mean"),
        sigma_tail5_mean=("sigma_tail5_mean_gate", "mean"),
    ).reset_index()
    print(agg.to_string(index=False))
PY

echo
echo "[TRACER] slippery micro-brake staged v4 done"
echo "[TRACER] joined csv: $JOINED_OUT_CSV"
